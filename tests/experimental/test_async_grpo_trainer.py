# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import itertools
import queue
from unittest.mock import patch

import numpy as np
import pytest
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.async_grpo_trainer import RolloutQueueDataset
from trl.experimental.async_grpo.async_rollout_worker import AsyncRolloutWorker, RolloutGroup, RolloutSample

from ..testing_utils import TrlTestCase


def dummy_reward_func(completions, **kwargs):
    return [float(hash(c[0]["content"]) % 100) / 100.0 for c in completions]


class _StubRolloutWorker:
    """Minimal rollout worker stub for testing the trainer in isolation."""

    def __init__(self, tokenizer, dataset, num_generations: int = 8, samples_per_weight_sync: int = 10):
        self.rollout_buffer = queue.Queue()
        self._samples_per_weight_sync = samples_per_weight_sync
        self._model_version = 0
        self._sample_iter = self._make_sample_iter(tokenizer, dataset, num_generations)

    def _make_sample_iter(self, tokenizer, dataset, num_generations):
        for row in itertools.cycle(dataset):
            completions = [
                [{"role": "assistant", "content": f"{row['completion'][0]['content']} {idx}"}]
                for idx in range(num_generations)
            ]
            prompt_completions = [row["prompt"] + completion for completion in completions]
            prompt_ids = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            prompt_completion_ids = tokenizer.apply_chat_template(
                prompt_completions, tokenize=True, add_generation_prompt=False, return_dict=False
            )
            rewards = np.array(dummy_reward_func(completions))
            advantages = (rewards - rewards.mean()) / rewards.std()
            for idx in range(num_generations):
                completion_ids = prompt_completion_ids[idx][len(prompt_ids) :]
                yield RolloutSample(
                    prompt=row["prompt"],
                    completion=completions[idx],
                    input_ids=prompt_ids + completion_ids,
                    completion_mask=[0] * len(prompt_ids) + [1] * len(completion_ids),
                    old_log_probs=[0.0] * len(prompt_ids) + [-0.5] * len(completion_ids),
                    advantage=float(advantages[idx]),
                    model_version=self._model_version,
                    metrics={"reward": float(rewards[idx]), "reward_std": float(rewards.std())},
                )

    def _fill_queue(self):
        for _ in range(self._samples_per_weight_sync):
            self.rollout_buffer.put(next(self._sample_iter))

    def start(self):
        self._fill_queue()

    def update_model_version(self, version):
        self._model_version = version
        self._fill_queue()

    def stop(self):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    def send_weights(self, iterator):
        pass


class TestAsyncGRPOTrainer(TrlTestCase):
    def test_init_minimal(self):
        # Test that AsyncGRPOTrainer can be instantiated with only model, reward_model and train_dataset
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
        )

    def test_train(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            vllm_server_timeout=5.0,  # short timeout so test fails fast if queue runs dry
            report_to="none",
        )
        trainer = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,  # unused: the stub pre-computes rewards, but the trainer requires this argument
            args=training_args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize("loss_type", ["grpo", "dapo"])
    def test_train_loss_types(self, loss_type):
        # Both loss types should train end-to-end and update parameters.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            vllm_server_timeout=5.0,
            loss_type=loss_type,
            report_to="none",
        )
        trainer = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            args=training_args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
        )
        previous = {n: p.clone() for n, p in trainer.model.named_parameters()}
        trainer.train()
        assert any(not torch.equal(previous[n], p) for n, p in trainer.model.named_parameters())

    def test_unknown_loss_type_raises(self):
        # AsyncGRPOConfig accepts arbitrary strings; the error surfaces in compute_loss.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            vllm_server_timeout=5.0,
            loss_type="not_a_real_loss",
            report_to="none",
        )
        trainer = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            args=training_args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
        )
        with pytest.raises(ValueError, match="Unknown loss type"):
            trainer.train()


def _make_sample(model_version=0, prompt_len=2, completion_len=3):
    """Build a RolloutSample with a known completion-token count."""
    return RolloutSample(
        prompt=[{"role": "user", "content": "x"}],
        completion=[{"role": "assistant", "content": "y"}],
        input_ids=[1] * (prompt_len + completion_len),
        completion_mask=[0] * prompt_len + [1] * completion_len,
        old_log_probs=[0.0] * (prompt_len + completion_len),
        advantage=0.0,
        model_version=model_version,
        metrics={"reward": 0.0},
    )


class TestRolloutQueueDataset:
    """Verify the pre-buffering logic that powers exact-DAPO normalization."""

    def test_accum_total_tokens_is_sum_of_completion_tokens(self):
        # Each sample has 4 completion tokens; with samples_per_accum_batch=6 → total = 24.
        q = queue.Queue()
        for _ in range(6):
            q.put(_make_sample(prompt_len=2, completion_len=4))
        ds = RolloutQueueDataset(
            rollout_queue=q,
            model_version_fn=lambda: 0,
            max_staleness=10,
            timeout=1.0,
            samples_per_accum_batch=6,
        )
        items = []
        for item in ds:
            items.append(item)
            if len(items) == 6:
                break
        assert len(items) == 6
        assert all(item["accum_total_tokens"] == 24 for item in items)

    def test_variable_completion_lengths_sum_correctly(self):
        # Lengths [3, 5, 2, 4] → total = 14.
        q = queue.Queue()
        for length in [3, 5, 2, 4]:
            q.put(_make_sample(prompt_len=1, completion_len=length))
        ds = RolloutQueueDataset(
            rollout_queue=q,
            model_version_fn=lambda: 0,
            max_staleness=10,
            timeout=1.0,
            samples_per_accum_batch=4,
        )
        items = []
        for item in ds:
            items.append(item)
            if len(items) == 4:
                break
        assert all(item["accum_total_tokens"] == 14 for item in items)

    def test_stale_samples_dropped_then_count_is_correct(self):
        # First sample is stale and should be dropped; total should reflect only the kept samples.
        q = queue.Queue()
        q.put(_make_sample(model_version=0, completion_len=10))  # stale (max_staleness=0, current_version=5)
        q.put(_make_sample(model_version=5, completion_len=3))
        q.put(_make_sample(model_version=5, completion_len=4))
        ds = RolloutQueueDataset(
            rollout_queue=q,
            model_version_fn=lambda: 5,
            max_staleness=0,
            timeout=1.0,
            samples_per_accum_batch=2,
        )
        items = list(itertools.islice(iter(ds), 2))
        assert len(items) == 2
        assert all(item["accum_total_tokens"] == 7 for item in items)

    def test_returns_on_timeout(self):
        # Empty queue → __iter__ should exit (StopIteration) after the timeout instead of hanging.
        ds = RolloutQueueDataset(
            rollout_queue=queue.Queue(),
            model_version_fn=lambda: 0,
            max_staleness=10,
            timeout=0.1,
            samples_per_accum_batch=2,
        )
        assert list(ds) == []


def _make_worker_for_scoring(scale_rewards="group", dynamic_sampling=False):
    """Bypass __init__ (which needs a live vLLM server) and set only what _score_group reads."""
    worker = AsyncRolloutWorker.__new__(AsyncRolloutWorker)
    worker.reward_funcs = [lambda completions, **kwargs: [1.0, 2.0, 3.0, 4.0]]  # placeholder, overridden per-test
    worker.reward_func_names = ["dummy"]
    worker.scale_rewards = scale_rewards
    worker.dynamic_sampling = dynamic_sampling
    worker._total_groups_dropped = 0
    return worker


def _make_group(num_completions=4):
    return RolloutGroup(
        prompt=[{"role": "user", "content": "x"}],
        prompt_ids=[1, 2],
        reward_kwargs={},
        completions=[[{"role": "assistant", "content": f"c{i}"}] for i in range(num_completions)],
        completions_ids=[[10 + i] for i in range(num_completions)],
        completions_logprobs=[[0.0] for _ in range(num_completions)],
        tool_mask=[[1] for _ in range(num_completions)],
        tool_call_counts=[0] * num_completions,
        tool_failure_counts=[0] * num_completions,
        model_version=0,
    )


class TestAsyncRolloutWorkerScoring:
    """Test the rollout worker scoring path (advantage scaling + dynamic sampling) in isolation."""

    def test_scale_rewards_group_divides_by_std(self):
        rewards = [1.0, 2.0, 3.0, 4.0]
        worker = _make_worker_for_scoring(scale_rewards="group")
        worker.reward_funcs = [lambda completions, **kwargs: rewards]
        group = _make_group(num_completions=4)
        samples = asyncio.run(worker._score_group(group))
        arr = np.array(rewards)
        expected = (arr - arr.mean()) / (arr.std() + 1e-8)
        assert np.allclose([s.advantage for s in samples], expected, atol=1e-5)

    def test_scale_rewards_none_skips_std(self):
        rewards = [1.0, 2.0, 3.0, 4.0]
        worker = _make_worker_for_scoring(scale_rewards="none")
        worker.reward_funcs = [lambda completions, **kwargs: rewards]
        group = _make_group(num_completions=4)
        samples = asyncio.run(worker._score_group(group))
        arr = np.array(rewards)
        expected = arr - arr.mean()  # Dr. GRPO: centered, no std divide
        assert np.allclose([s.advantage for s in samples], expected, atol=1e-5)

    def test_scale_rewards_batch_skips_std_in_worker(self):
        # 'batch' is centered in the worker; the trainer applies the cross-rank std divide later.
        rewards = [1.0, 2.0, 3.0, 4.0]
        worker = _make_worker_for_scoring(scale_rewards="batch")
        worker.reward_funcs = [lambda completions, **kwargs: rewards]
        group = _make_group(num_completions=4)
        samples = asyncio.run(worker._score_group(group))
        arr = np.array(rewards)
        expected = arr - arr.mean()
        assert np.allclose([s.advantage for s in samples], expected, atol=1e-5)

    def test_dynamic_sampling_drops_zero_std_group(self):
        # All completions get the same reward → std=0 → group should be dropped.
        worker = _make_worker_for_scoring(scale_rewards="group", dynamic_sampling=True)
        worker.reward_funcs = [lambda completions, **kwargs: [0.5, 0.5, 0.5, 0.5]]
        group = _make_group(num_completions=4)
        samples = asyncio.run(worker._score_group(group))
        assert samples == []
        assert worker._total_groups_dropped == 1

    def test_dynamic_sampling_keeps_nonzero_std_group(self):
        worker = _make_worker_for_scoring(scale_rewards="group", dynamic_sampling=True)
        worker.reward_funcs = [lambda completions, **kwargs: [0.0, 1.0, 0.0, 1.0]]
        group = _make_group(num_completions=4)
        samples = asyncio.run(worker._score_group(group))
        assert len(samples) == 4
        assert worker._total_groups_dropped == 0

    def test_dynamic_sampling_off_keeps_zero_std_group(self):
        worker = _make_worker_for_scoring(scale_rewards="group", dynamic_sampling=False)
        worker.reward_funcs = [lambda completions, **kwargs: [0.5, 0.5, 0.5, 0.5]]
        group = _make_group(num_completions=4)
        samples = asyncio.run(worker._score_group(group))
        assert len(samples) == 4

    def test_invalid_scale_rewards_raises(self):
        # Exercise the real __init__ validation by patching out the vLLM-dependent setup steps.
        with (
            patch.object(AsyncRolloutWorker, "_wait_for_server_ready_sync"),
            patch.object(AsyncRolloutWorker, "_init_weight_transfer"),
            patch(
                "trl.experimental.async_grpo.async_rollout_worker.is_vllm_available",
                return_value=True,
            ),
            patch(
                "trl.experimental.async_grpo.async_rollout_worker.add_response_schema",
                side_effect=lambda tok: tok,
            ),
            patch(
                "trl.experimental.async_grpo.async_rollout_worker.is_chat_template_prefix_preserving",
                return_value=True,
            ),
        ):
            tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
            with pytest.raises(ValueError, match="Unknown scale_rewards"):
                AsyncRolloutWorker(
                    model_name="x",
                    dataset=None,
                    reward_funcs=[lambda **k: [0.0]],
                    processing_class=tokenizer,
                    max_inflight_tasks=1,
                    scale_rewards="weird",
                )
