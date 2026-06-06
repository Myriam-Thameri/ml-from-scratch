# Optimizers

This folder contains optimizer implementations for updating model parameters during training.

## Overview

Each optimizer implements the `Optimizer` base class from `base.py` and provides an `update` method that modifies weights (`w`) and bias (`b`) based on gradients (`dw`, `db`). The `iteration` parameter is used by some optimizers to adjust the learning rate over time.

## Files

- `base.py`
  - Defines the `Optimizer` base class.
  - Declares `update(self, w, b, dw, db, iteration)` as the required interface.

- `sgd.py`
  - Implements `SGDD`, a standard stochastic gradient descent optimizer.
  - Uses a fixed learning rate `lr` to update parameters.

- `adaptive.py`
  - Implements `AdaptiveSGD`, a simple adaptive learning rate optimizer.
  - Decreases the learning rate when the provided loss increases compared to the previous step.
  - Useful for basic training heuristics that reduce step size after a bad update.

- `decay.py`
  - Implements `DecaySGD`, a learning rate decay optimizer.
  - Scales the base learning rate by `1 / sqrt(iteration + 1)`.
  - Helps reduce update magnitude as training progresses.

## Optimizers explained

### `SGDD`

- A plain SGD optimizer.
- Parameters:
  - `lr`: learning rate (default `0.01`).
- Update rule:
  - `w -= lr * dw`
  - `b -= lr * db`

### `AdaptiveSGD`

- A simple adaptive optimizer.
- Parameters:
  - `lr`: base learning rate (default `0.01`).
- Behavior:
  - If the current loss is higher than the previous loss, the optimizer reduces the step size by multiplying `lr` by `0.05`.
  - Stores the previous loss in `self.prev_loss`.
- Update rule:
  - `w -= lr * dw`
  - `b -= lr * db`

### `DecaySGD`

- SGD with learning rate decay.
- Parameters:
  - `lr`: base learning rate (default `0.01`).
- Learning rate schedule:
  - `lr_t = lr / sqrt(iteration + 1)`
- Update rule:
  - `w -= lr_t * dw`
  - `b -= lr_t * db`

## Usage

Import and instantiate the optimizer you want to use, then call `update` on each training step.

```python
from optimizers.sgd import SGDD

opt = SGDD(lr=0.01)
w, b = opt.update(w, b, dw, db, iteration)
```

For `AdaptiveSGD`, pass the current loss when available:

```python
from optimizers.adaptive import AdaptiveSGD

opt = AdaptiveSGD(lr=0.01)
w, b = opt.update(w, b, dw, db, loss=current_loss)
```

For `DecaySGD`:

```python
from optimizers.decay import DecaySGD

opt = DecaySGD(lr=0.01)
w, b = opt.update(w, b, dw, db, iteration)
```
