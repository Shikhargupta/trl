# TRL (fork)

A fork of [🤗 TRL](https://github.com/huggingface/trl) with additional post-training features.

## Features on top of standard TRL

- **`AsyncGRPOTrainer`** — asynchronous GRPO where generation is offloaded to an external vLLM server running alongside training, decoupling rollout from the gradient update loop. Supports `loss_type` (including DAPO), `scale_rewards`, dynamic sampling, and final logits softcapping.

```python
from trl.experimental.async_grpo import AsyncGRPOTrainer
from trl.rewards import accuracy_reward
from datasets import load_dataset

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = AsyncGRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

## Installation

```bash
pip install -e .
```

Everything else follows upstream TRL — see the [upstream README](https://github.com/huggingface/trl) and [documentation](https://huggingface.co/docs/trl/index).

## License

Apache-2.0. See [LICENSE](LICENSE).
