import numpy as np
import celer
from datetime import datetime, timedelta
from typing import List
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def moving_average_smoother(signal, window_length = 7):
    signal_padded = np.append(np.nan * np.ones(window_length - 1), signal)
    signal_smoothed = (np.convolve(signal_padded, np.ones(window_length, dtype=int), mode="valid")/ window_length)
    return signal_smoothed

def standardize_X(X):
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1
    standardized_X = (X - X_mean) / X_std
    standardized_X[np.isnan(standardized_X)] = 0
    return standardized_X, X_mean, X_std

def standardize_y(y):
    y_mean, y_std = y.mean(), y.std()
    standardized_y = (y - y_mean) / y_std
    standardized_y[np.isnan(standardized_y)] = 0
    return standardized_y, y_mean, y_std

def recover_coefficients(coef, X_std, y_std):
    return (coef / X_std) * y_std

def recover_intercept(X, y, coef, intercept_length):
    intercept = y - np.dot(X, coef)
    return np.mean(intercept[-intercept_length:])

def group_X(X, lag_set):
    grouped_X = np.array([X[r_idx - lag_set, :].T.flatten() for r_idx in range(max(lag_set), X.shape[0])])
    return grouped_X

def extend_indices(indices, group_size):
    extended_indices = []
    for idx in indices:
        extended_indices.extend(list(range(idx * group_size, idx * group_size + group_size)))
    return extended_indices

def shrink_indices(extended_indices, group_size):
    indices = [extended_indices[i] // group_size for i in range(0, len(extended_indices), group_size)]
    return indices

def date_before(date_int, diff_days):
    date_obj = datetime.strptime(str(date_int), '%Y%m%d')
    new_date_obj = date_obj - timedelta(days=int(diff_days))
    new_date_int = int(new_date_obj.strftime('%Y%m%d'))
    return new_date_int

def date_after(date_int, diff_days):
    date_obj = datetime.strptime(str(date_int), '%Y%m%d')
    new_date_obj = date_obj + timedelta(days=int(diff_days))
    new_date_int = int(new_date_obj.strftime('%Y%m%d'))
    return new_date_int
