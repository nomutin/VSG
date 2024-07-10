# VSG

![python](https://img.shields.io/badge/python-3.10-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Lightning implementation of [VSG](https://arxiv.org/abs/2210.11698).

## API

```python
from vsg import VSG

reference = "wandb reference"
vsg = vsg.load_from_wandb(reference=reference)
```

## References

### Paper

- [Learning Robust Dynamics through Variational Sparse Gating [Jain+ 2022]](https://arxiv.org/abs/2210.11698)
- [Mastering Atari with Discrete World Models [Hafner+ 2021]](https://arxiv.org/abs/2010.02193)

### Code

- [google-research/dreamer](https://github.com/google-research/dreamer)
- [danijar/dreamerv2](https://github.com/danijar/dreamerv2)
- [arnavkj1995/VSG](https://github.com/arnavkj1995/VSG)
