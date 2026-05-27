def mean_absolute_error(y_true, y_pred):
    return sum(abs(y_true - y_pred)) / len(y_true)
