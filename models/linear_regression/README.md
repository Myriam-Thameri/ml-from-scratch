# Linear Regression From Scratch

This folder contains a NumPy-based implementation of **Linear Regression** with support for feature scaling, regularization, and an adaptive gradient optimizer.

The implementation is designed to demonstrate:
* the linear regression model and its objective
* why z-score scaling helps training
* how gradient updates are computed
* how adaptive optimization improves convergence
* how early stopping avoids overfitting and wasted iterations

---

## 1. Model Overview

Linear regression models the relationship between features and a continuous output using a linear equation:

$$\hat{y} = Xw + b$$

Where:
* $X \in \mathbb{R}^{n \times d}$ is the feature matrix
* $w \in \mathbb{R}^{d}$ is the vector of weights
* $b \in \mathbb{R}$ is the bias term
* $\hat{y} \in \mathbb{R}^{n}$ are the predictions

For one sample, the prediction is:

$$\hat{y}^{(i)} = w_1 x^{(i)}_1 + w_2 x^{(i)}_2 + \dots + w_d x^{(i)}_d + b$$

---

## 2. Loss Function: Mean Squared Error

The training objective is to minimize the Mean Squared Error (MSE):

$$L(w, b) = \frac{1}{n} \sum_{i=1}^{n} \left(y^{(i)} - \hat{y}^{(i)}\right)^2$$

This loss measures the average squared difference between true targets $y$ and predictions $\hat{y}$.

### Why MSE?
* It penalizes large errors more heavily than small errors.
* It is differentiable, enabling gradient-based optimization.
* It provides a smooth surface for finding the optimal parameters.

---

## 3. Feature Scaling: Z-score Normalization

Before optimization, features are standardized using z-score:

$$X^{(i)}_{\text{scaled}} = \frac{X^{(i)} - \mu}{\sigma}$$

Where:
* $\mu$ is the mean of each feature across the training set
* $\sigma$ is the standard deviation of each feature

The code also safeguards constant features by replacing zero standard deviation with 1.

### Why normalize?
* Prevents features with large magnitudes from dominating learning.
* Makes gradient descent more stable and faster.
* Keeps all feature dimensions on a similar scale.

---

## 4. Gradients and Parameter Updates

The gradient of the MSE loss with respect to the weights is:

$$\frac{\partial L}{\partial w} = \frac{1}{n} X^T (Xw + b - y)$$

The gradient with respect to the bias is:

$$\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (X^{(i)}w + b - y^{(i)})$$

In code, these become:

* $dw = \frac{1}{n} X^T \text{error}$
* $db = \frac{1}{n} \sum \text{error}$

Where $\text{error} = y_{\text{pred}} - y$.

---

## 5. Regularization Support

This implementation supports optional weight regularization during training:

* L2 regularization (Ridge):

$$\text{loss} = \text{MSE} + \lambda \|w\|_2^2$$

$$dw = \frac{1}{n} X^T \text{error} + 2 \lambda w$$

* L1 regularization (Lasso):

$$\text{loss} = \text{MSE} + \lambda \|w\|_1$$

$$dw = \frac{1}{n} X^T \text{error} + \lambda \, \text{sign}(w)$$

Regularization helps prevent overfitting by penalizing large weight values.

---

## 6. Adaptive Optimization

The model uses an adaptive optimizer from `optimizers.adaptive.AdaptiveSGD`.

Instead of updating weights with a fixed learning rate, the optimizer adjusts the update step based on the current gradient and training iteration.

This can improve convergence speed and reduce the need for manual learning rate tuning.

---

## 7. Early Stopping and Convergence

Training checks the validation of loss improvement using:
* `toleration`: minimum required loss decrease
* `patience`: number of iterations without sufficient improvement

If the loss does not decrease by more than `toleration` for `patience` consecutive iterations, training stops early.

This prevents unnecessary iterations once the model has effectively converged.

---

## 8. Workflow of `LinearRegression`

### `fit(X, y, regularizer=None, lambda_=0.01)`

1. Validates input shapes:
   * `X` must be 2D
   * `y` must be 1D
   * sample counts of `X` and `y` must match
2. Initializes weights and bias
3. Standardizes training features
4. Computes predictions and loss each iteration
5. Applies optional regularization
6. Computes gradients and updates parameters via the optimizer
7. Tracks loss history in `self.losses`
8. Stops early if training converges

### `predict(X)`

1. Applies the same z-score scaling using training mean and std
2. Computes predictions with:

$$\hat{y} = X_{\text{scaled}} w + b$$

3. Returns the continuous prediction values

---

## 9. Equations in Code

Prediction:

$$y_{\text{pred}} = Xw + b$$

Loss:

$$L = \frac{1}{n} \sum (y_{\text{pred}} - y)^2$$

Weight gradient:

$$dw = \frac{1}{n} X^T (y_{\text{pred}} - y)$$

Bias gradient:

$$db = \frac{1}{n} \sum (y_{\text{pred}} - y)$$

Parameter update:

$$w \leftarrow w - \alpha \, dw$$

$$b \leftarrow b - \alpha \, db$$

---

## 10. Usage Example

```python
from models.linear_regression.model import LinearRegression

model = LinearRegression(learning_rate=0.01, verbose=True, max_iters=100000, patience=10, toleration=1e-6)
model.fit(X_train, y_train, regularizer='L2', lambda_=0.01)
y_pred = model.predict(X_test)
```

Inspect training progress with `model.losses` to verify convergence.

---

## 11. Practical Notes

* `AdaptiveSGD` is used for parameter updates, so learning behavior may differ from simple batch gradient descent.
* The model stores the mean and standard deviation from training and reuses them for prediction.
* Regularization is optional and only applies to weight updates, not bias.

---

## 12. Key Insights

* Linear regression is a linear model that minimizes squared error.
* Scaling features before training is essential for stable gradient-based optimization.
* Adaptive optimization helps make learning more robust.
* Regularization can improve generalization by limiting weight magnitude.
* Early stopping helps prevent wasted training after convergence.
