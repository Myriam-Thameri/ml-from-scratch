import numpy as np

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true-y_pred) ** 2)

#How wrong my model is
def RSS(y_true,y_pred):
    return np.sum((y_true-y_pred)**2)

def SSR(y_true,y_pred):
    y_mean = np.mean(y_true)
    return np.sum((y_pred - y_mean) ** 2)

def TSS(y_true, y_pred):
    return RSS(y_true,y_pred) + SSR(y_true, y_pred)

def r_squared(y_true,y_pred):
    return 1 - (RSS(y_true,y_pred)/TSS(y_true,y_pred))