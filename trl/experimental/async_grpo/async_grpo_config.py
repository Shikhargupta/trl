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

from dataclasses import dataclass, field

from trl.trainer.base_config import _BaseConfig


@dataclass
class AsyncGRPOConfig(_BaseConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`AsyncGRPOTrainer`].

    This class includes only the parameters that are specific to asynchronous GRPO training. For a full list of
    training arguments, please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values
    in this class may differ from those in [`~transformers.TrainingArguments`].

    Parameters:
        > Parameters that control generation

        num_generations (`int`, *optional*, defaults to `8`):
            Number of generations per prompt to sample.
        max_completion_length (`int`, *optional*, defaults to `2048`):
            Maximum number of tokens to generate per completion.
        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        chat_template_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` function when generating completions.
        max_tool_calling_iterations (`int`, *optional*):
            Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and generation
            stops when the model generates a response turn with no tool calls or when the total response length reaches
            `max_completion_length`.

        > Parameters that control the vLLM server

        vllm_server_base_url (`str`, *optional*, defaults to `"http://localhost:8000"`):
            Base URL of the vLLM server used for generation (e.g., `"http://localhost:8000"`).
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Total timeout duration in seconds to wait for the vLLM server to be ready.
        request_timeout (`int`, *optional*, defaults to `600`):
            Timeout in seconds for individual HTTP requests to the vLLM server.

        > Parameters that control the training

        epsilon (`float`, *optional*, defaults to `0.2`):
            Lower-bound epsilon value for clipping.
        epsilon_high (`float`, *optional*, defaults to `0.2`):
            Upper-bound epsilon value for clipping.
        loss_type (`str`, *optional*, defaults to `"grpo"`):
            Loss formulation. Supported values are:

            - `"grpo"` (default): Normalizes the summed per-token loss by the number of completion tokens in the
              microbatch. Tokens in long-sequence microbatches are down-weighted relative to tokens in short-sequence
              microbatches.
            - `"dapo"`: Normalizes the summed per-token loss by the per-microbatch sequence count, so that every token
              contributes equally across the full accumulated batch. This is the formulation from the
              [DAPO](https://huggingface.co/papers/2503.14476) paper.
        scale_rewards (`str`, *optional*, defaults to `"group"`):
            Reward scaling strategy used when computing advantages:

            - `"group"` (default): Scale each reward by the standard deviation within its group (the rollouts for
              one prompt), ensuring unit variance per group. This is the standard GRPO formulation.
            - `"batch"`: Center each reward by its group mean, then divide by the standard deviation across all
              rollouts in the trainer microbatch (gathered across DDP ranks).
            - `"none"`: Center each reward by its group mean with no std scaling. Recommended by the
              [Dr. GRPO](https://huggingface.co/papers/2503.20783) paper, which shows that std-based scaling
              introduces a question-level difficulty bias. Note that turning off std-based scaling also removes
              variance normalization, so update magnitudes depend directly on the raw reward scale and batch
              composition.
        dynamic_sampling (`bool`, *optional*, defaults to `False`):
            Whether to enable dynamic sampling, as proposed in the [DAPO](https://huggingface.co/papers/2503.14476)
            paper. When enabled, groups whose completions all receive the same reward (i.e. zero-std groups, which
            would produce zero advantages) are discarded by the rollout worker instead of being pushed to the trainer.
            The worker keeps generating until non-degenerate groups are produced.

        > Parameters that control the async rollout pipeline

        max_inflight_tasks (`int`, *optional*, defaults to `-1`):
            Maximum number of concurrent generation tasks sent to the vLLM server. Defaults to `-1` (auto), which
            sets it to `max_staleness * per_device_train_batch_size * gradient_accumulation_steps * num_processes`.
            If using tool-use environments, you may want to set this manually based on how many parallel environments
            you can run.
        max_staleness (`int`, *optional*, defaults to `4`):
            Maximum number of weight update steps a rollout sample can lag behind the current model version before
            being discarded.
        queue_maxsize (`int`, *optional*, defaults to `1024`):
            Maximum number of rollout samples to buffer in the rollout queue.
        weight_sync_steps (`int`, *optional*, defaults to `1`):
            Number of training steps between weight synchronizations to the vLLM server.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps.
        num_completions_to_print (`int`, *optional*, defaults to `3`):
            Number of completions to print when `log_completions=True`.

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `1e-6` instead of `5e-5`.
    """

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    logging_steps: float = field(
        default=1,
        metadata={
            "help": "Log every X update steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )

    # Parameters that control generation
    num_generations: int = field(
        default=8,
        metadata={"help": "Number of generations per prompt to sample."},
    )
    max_completion_length: int = field(
        default=2048,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    chat_template_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to the `apply_chat_template` function when generating "
            "completions."
        },
    )
    max_tool_calling_iterations: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and "
            "generation stops when the model generates a response turn with no tool calls or when the total response "
            "length reaches `max_completion_length`."
        },
    )

    # Parameters that control the vLLM server
    vllm_server_base_url: str = field(
        default="http://localhost:8000",
        metadata={"help": "Base URL of the vLLM server used for generation (e.g., 'http://localhost:8000')."},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be ready. If the server is not "
            "up after the timeout, a `TimeoutError` is raised."
        },
    )
    request_timeout: int = field(
        default=600,
        metadata={"help": "Timeout in seconds for individual HTTP requests to the vLLM server."},
    )

    # Parameters that control the training
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Lower-bound epsilon value for clipping."},
    )
    epsilon_high: float = field(
        default=0.2,
        metadata={"help": "Upper-bound epsilon value for clipping."},
    )
    loss_type: str = field(
        default="grpo",
        metadata={
            "help": "Loss formulation. Supported values are 'grpo' and 'dapo'. 'grpo' (default) normalizes the "
            "summed per-token loss by the number of completion tokens in the microbatch — long-sequence microbatches "
            "down-weight their own tokens. 'dapo' normalizes by the per-microbatch sequence count instead, so that "
            "every token contributes equally across the full accumulated batch (the formulation from "
            "https://huggingface.co/papers/2503.14476).",
            "choices": ["grpo", "dapo"],
        },
    )
    scale_rewards: str = field(
        default="group",
        metadata={
            "help": "Reward scaling strategy used when computing advantages. Supported values: "
            "'group' (default): scale each reward by the standard deviation within its group (the rollouts for one "
            "prompt), ensuring unit variance per group. 'batch': center per-group, then divide by the standard "
            "deviation across all rollouts in the trainer microbatch (gathered across DDP ranks). 'none': center "
            "per-group with no std scaling, as recommended in the Dr. GRPO paper "
            "(https://huggingface.co/papers/2503.20783), which shows that std-based scaling introduces a "
            "question-level difficulty bias.",
            "choices": ["group", "batch", "none"],
        },
    )
    dynamic_sampling: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable dynamic sampling, as proposed in the DAPO paper "
            "(https://huggingface.co/papers/2503.14476). When enabled, groups whose completions all receive the same "
            "reward (i.e. zero-std groups, which produce zero advantages) are discarded by the rollout worker instead "
            "of being pushed to the trainer. The worker keeps generating until non-degenerate groups are produced."
        },
    )

    # Parameters that control the async rollout pipeline
    max_inflight_tasks: int = field(
        default=-1,
        metadata={
            "help": "Maximum number of concurrent generation tasks sent to the vLLM server. Defaults to -1 (auto), "
            "which sets it to `max_staleness * per_device_train_batch_size * gradient_accumulation_steps * "
            "num_processes`. Generating more samples than this is wasteful since they will be discarded as stale "
            "before the trainer can consume them. If using tool-use environments, you may want to set this manually "
            "based on how many parallel environments you can run."
        },
    )
    max_staleness: int = field(
        default=4,
        metadata={
            "help": "Maximum number of weight update steps a rollout sample can lag behind the current model version "
            "before being discarded."
        },
    )
    queue_maxsize: int = field(
        default=1024,
        metadata={"help": "Maximum number of rollout samples to buffer in the rollout queue."},
    )
    weight_sync_steps: int = field(
        default=1,
        metadata={"help": "Number of training steps between weight synchronizations to the vLLM server."},
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )
    num_completions_to_print: int = field(
        default=3,
        metadata={"help": "Number of completions to print when `log_completions=True`."},
    )

    def __post_init__(self):
        super().__post_init__()

        # Accelerator config: required for the async IterableDataset-backed dataloader to work correctly.
        # split_batches=True and dispatch_batches=True ensure that the main process drives the dataloader
        # and batches are broadcast to other processes rather than each process pulling independently.
        if not hasattr(self, "accelerator_config") or self.accelerator_config is None:
            self.accelerator_config = {"split_batches": True, "dispatch_batches": True}
        elif isinstance(self.accelerator_config, dict):
            self.accelerator_config["split_batches"] = True
            self.accelerator_config["dispatch_batches"] = True
        else:
            self.accelerator_config.split_batches = True
            self.accelerator_config.dispatch_batches = True
