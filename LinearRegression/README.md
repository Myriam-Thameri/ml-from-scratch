# Linear Regression From Scratch

This project implements **Linear Regression from scratch using NumPy**, without using any machine learning libraries like scikit-learn.

The goal is to understand:
* how linear regression works mathematically
* how gradient descent optimizes parameters
* how feature normalization affects training
* how predictions are made from learned weights

---

## 1. Model Intuition

Linear Regression tries to model a relationship between inputs and outputs using a linear equation:

$$\hat{y} = Xw + b$$

Where:
* $X$: input features (matrix)
* $w$: weights (learned parameters)
* $b$: bias (offset term)
* $\hat{y}$: predicted output

Each feature contributes to the prediction:

$$\hat{y} = w_1x_1 + w_2x_2 + \cdots + b$$

* weights control importance of each feature
* bias shifts prediction up/down

---

## 2. Training Objective (Loss Function)

We want predictions to be as close as possible to true values.
We use **Mean Squared Error (MSE)**:

$$L = \frac{1}{n} \sum (y_{\text{pred}} - y)^2$$

Where:
* $n$: number of samples
* error is squared to penalize large mistakes more

### Why squared error?
* prevents cancellation of positive/negative errors
* penalizes large mistakes more strongly
* creates a smooth differentiable function (needed for gradient descent)

---

## 3. Optimization: Gradient Descent

We minimize the loss by updating parameters iteratively:

$$w = w - \alpha \frac{\partial L}{\partial w}$$

$$b = b - \alpha \frac{\partial L}{\partial b}$$

Where:
* $\alpha$: learning rate

---

## 4. Gradient Derivation (Core Idea)

We compute how each parameter affects the loss.

### Prediction:
$$y_{\text{pred}} = Xw + b$$

### Error:
$$\text{error} = y_{\text{pred}} - y$$

---

### Gradient for weights:

$$\frac{\partial L}{\partial w} = \frac{1}{n} X^T (y_{\text{pred}} - y)$$

**Intuition:**
* $X^T$: selects each feature across all samples
* error: tells how wrong predictions are
* result: how much each feature contributed to error

---

### Gradient for bias:

$$\frac{\partial L}{\partial b} = \frac{1}{n} \sum (y_{\text{pred}} - y)$$

**Intuition:**
Bias affects all predictions equally, so we average the error.

---

## 5. Feature Normalization (Z-score)

Before training, features are normalized:

$$X = \frac{X - \mu}{\sigma}$$

Where:
* $\mu$: mean of each feature
* $\sigma$: standard deviation

### Why normalization is needed

Without normalization:
* features with large values dominate learning
* gradient descent becomes unstable or slow

With normalization:
* all features are on the same scale
* training becomes faster and stable

---

## 6. Training Process (fit function)

The model trains using iterative updates:

### Step 1: Predict
$$y_{\text{pred}} = Xw + b$$

### Step 2: Compute error
$$\text{error} = y_{\text{pred}} - y$$

### Step 3: Compute gradients
$$dw = \frac{1}{n} X^T \text{ error}$$

$$db = \frac{1}{n} \sum \text{error}$$

### Step 4: Update parameters
$$w = w - \alpha \cdot dw$$

$$b = b - \alpha \cdot db$$

---

## 7. Loss Tracking

We store loss at each iteration:

$$L = \frac{1}{n} \sum (\text{error})^2$$

This helps visualize:
* convergence speed
* training stability
* whether model is learning properly

---

## 8. Prediction (Inference)

After training:

$$\hat{y} = Xw + b$$

Important:
* same normalization must be applied to input data
* uses learned weights only (no training)

---

## 9. Full Training Loop Summary

1. Normalize data
2. Initialize weights and bias
3. Repeat for N iterations:
   * predict
   * compute error
   * compute gradients
   * update parameters
   * store loss

---

## 10. Key Takeaways

* Linear regression learns a linear relationship between inputs and outputs
* Gradient descent minimizes prediction error step-by-step
* Feature scaling is essential for stable training
* All learning comes from repeated correction of errors

---

## Final Insight

Linear Regression is not just a formula:

> **It is an optimization process that gradually adjusts weights to reduce prediction error using gradients computed from data.**
