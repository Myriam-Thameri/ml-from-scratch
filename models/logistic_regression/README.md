# Logistic Regression From Scratch

This folder contains a simple implementation of **Logistic Regression** using only NumPy, without any external machine learning library.

The goal is to explain:
* how logistic regression models binary classification
* why the sigmoid function is used
* what binary cross-entropy loss measures
* how gradient descent updates weights
* why feature scaling is important

---

## 1. What is Logistic Regression?

Logistic Regression models the probability that a sample belongs to class 1 (positive) or class 0 (negative).

Instead of predicting a continuous number like linear regression, logistic regression predicts a probability:

$$\hat{y} = \sigma(z)$$

where:
* $z = Xw + b$ is the linear output
* $\sigma$ is the sigmoid function
* $\hat{y}$ is the predicted probability for class 1

The final class prediction is:

$$y_{\text{pred}} = \begin{cases}1 & \text{if } \hat{y} \ge 0.5 \\ 0 & \text{otherwise} \end{cases}$$

---

## 2. Sigmoid Function

The sigmoid function converts any real-valued number into a value between 0 and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Properties:
* when $z \to +\infty$, $\sigma(z) \to 1$
* when $z \to -\infty$, $\sigma(z) \to 0$
* when $z = 0$, $\sigma(z) = 0.5$

This makes sigmoid ideal for probability estimation.

---

## 3. Prediction Process

A logistic regression prediction is computed in two steps:

1. Linear combination of input features:

$$z = Xw + b$$

2. Convert linear output to probability:

$$\hat{y} = \sigma(z)$$

Then the model returns class 1 when the probability is at least 0.5.

---

## 4. Loss Function: Binary Cross-Entropy

To measure how well the model predicts probabilities, we use **binary cross-entropy** (also known as log loss):

$$L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

Where:
* $n$ is the number of samples
* $y^{(i)}$ is the true label (0 or 1)
* $\hat{y}^{(i)}$ is the predicted probability for sample $i$

Why this loss is used:
* it penalizes confident wrong predictions heavily
* it is differentiable, which enables gradient-based optimization
* it aligns with the maximum likelihood principle for binary classification

A small numerical constant $\varepsilon$ is added inside the log to avoid taking $\log(0)$.

---

## 5. Feature Scaling (Z-score Normalization)

Before training, features are scaled using the z-score formula:

$$X_{\text{scaled}} = \frac{X - \mu}{\sigma}$$

Where:
* $\mu$ is the mean of each feature across the training set
* $\sigma$ is the standard deviation of each feature

Why scaling helps:
* prevents features with large numeric ranges from dominating gradients
* makes gradient descent converge faster
* keeps training stable when multiple features have different scales

Note: The same mean and standard deviation from training are reused for prediction.

---

## 6. Gradient Descent Training

The model learns weights $w$ and bias $b$ by minimizing cross-entropy loss with gradient descent.

### Gradients
The gradients of the loss with respect to the parameters are:

$$dw = \frac{1}{n} X^T (\hat{y} - y)$$

$$db = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})$$

Intuition:
* $\hat{y} - y$ is the prediction error for each sample
* $X^T$ accumulates how each feature contributed to that error
* dividing by $n$ averages the contribution over all samples

### Parameter updates
Weights and bias are updated by taking a step in the opposite direction of the gradient:

$$w \leftarrow w - \alpha \, dw$$

$$b \leftarrow b - \alpha \, db$$

Where $\alpha$ is the learning rate.

---

## 7. Convergence and Early Stopping

Training stops when the loss stops improving.

This implementation tracks the best loss so far and uses two controls:
* `tolerance`: minimum required improvement in loss
* `patience`: number of iterations allowed without meaningful improvement

If the loss does not decrease by more than `tolerance` for more than `patience` iterations, training stops early.

---

## 8. Model Workflow

### `fit(X, y)`
* validates that $X$ and $y$ have the same number of samples
* initializes weights $w$ and bias $b$
* scales features using mean and standard deviation
* iteratively computes predictions, loss, gradients, and updates
* records the loss history in `self.losses`

### `predict(X)`
* applies the same scaling used during training
* computes linear output and sigmoid probability
* returns binary class labels using threshold 0.5

---

## 9. Practical Notes

* The current implementation includes optional `regularizer` and `lambda_` arguments in `fit`, but regularization is not applied yet.
* The bias `b` is updated separately from weights and does not require feature scaling.
* The sigmoid output is clipped internally during loss computation to avoid numerical issues.

---

## 10. Key Takeaways

* Logistic Regression is a linear classifier that outputs probabilities.
* The sigmoid function maps raw scores to [0, 1].
* Binary cross-entropy measures how well predicted probabilities match true labels.
* Gradient descent updates parameters to reduce the loss gradually.
* Feature scaling is essential for stable and efficient training.

---

## 11. Recommended Usage

1. Prepare your dataset with features `X` and binary labels `y`.
2. Create and fit the model:

```python
from models.logistic_regression.model import LogisticRegression

model = LogisticRegression(learning_rate=0.01, max_iters=100000, patience=10, tolerance=1e-6)
model.fit(X_train, y_train)
```

3. Predict new data:

```python
y_pred = model.predict(X_test)
```

4. Inspect training progress with `model.losses`.
