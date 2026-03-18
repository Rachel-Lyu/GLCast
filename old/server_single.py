import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import celer
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from knockpy import KnockoffFilter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

def moving_average_smoother(signal, window_length = 7):
    signal_padded = np.append(np.nan * np.ones(window_length - 1), signal)
    signal_smoothed = (np.convolve(signal_padded, np.ones(window_length, dtype=int), mode="valid")/ window_length)
    return signal_smoothed

def standardize_X(X):
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1
    standardized_X = (X - X_mean) / X_std
    standardized_X[np.isnan(standardized_X)] = 0
    return standardized_X, X_std

def standardize_y(y):
    y_mean, y_std = y.mean(), y.std()
    standardized_y = (y - y_mean) / y_std
    standardized_y[np.isnan(standardized_y)] = 0
    return standardized_y, y_std

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

# HealthDataHandler(state_list, target_data, state_data_dict)
# both target_data and state_data_dict should be smoothed
class HealthDataHandler:
    def __init__(self, state_list: List[str], target_data: pd.DataFrame, state_data: Dict[str, pd.DataFrame]):
        self.state_list = state_list
        self.target_data = target_data
        self.state_data = state_data

    def get_X_y(self, start_date: int, end_date: int, codes: List[str]):
        self.X = {}
        self.standardized_X = {}
        self.X_std = {}
        self.y = {}
        self.standardized_y = {}
        self.y_std = {}
        self.standardized_grouped_X = {}
        self.grouped_X = {}
        for state in self.state_list:
            X = self.state_data[state].loc[date_before(start_date, max(lag_set)):end_date][codes].to_numpy()
            y = self.target_data.loc[start_date:end_date, state].to_numpy()
            standardized_X, X_std = standardize_X(X)
            standardized_y, y_std = standardize_y(y)
            standardized_grouped_X = group_X(standardized_X, lag_set=lag_set)
            grouped_X = group_X(X, lag_set=lag_set)
            X = X[-(len(y)):]
            standardized_X = standardized_X[-(len(y)):]
            self.X[state] = X
            self.standardized_X[state] = standardized_X
            self.X_std[state] = X_std
            self.y[state] = y
            self.standardized_y[state] = standardized_y
            self.y_std[state] = y_std
            self.standardized_grouped_X[state] = standardized_grouped_X
            self.grouped_X[state] = grouped_X
        self.nS = len(y)
        self.nG = X.shape[1]
        self.nV = grouped_X.shape[1]

    def run_all_single(self, model, group_size, intercept_length, alpha = 0.3, refit_prev = False, states_prev_g = None, maxsteps = 5, states_prev_W = None, tau = 0.5, M = 10, M_max = 30, fdr = 0.1):
        # grpLasso derandomKnock forward
        ### TODO ###
        ### coef_method: ['standardized', 'original'] ###
        self.active_g_indices = {}
        self.recovered_coef = {}
        self.recovered_intercept = {}
        if model == 'grpLasso':
            self.unit_coef = {}
            for state in self.state_list:
                if refit_prev:
                    if states_prev_g is None:
                        prev_g_indices = None
                    else:
                        prev_g_indices = states_prev_g[state]
                else:
                    prev_g_indices = None
                print(state, end = ':')
                active_g_indices, unit_coef, recovered_coef, recovered_intercept = SIS_grpLasso_KKT(self.X[state], self.y[state], self.standardized_grouped_X[state], self.standardized_y[state], self.grouped_X[state], self.X_std[state], self.y_std[state], group_size, intercept_length, alpha = alpha, refit_prev = refit_prev, prev_g_indices = prev_g_indices)
                # model_coef = unit_coef if coef_method == 'standardized' else recovered_coef
                self.active_g_indices[state] = active_g_indices
                self.recovered_coef[state] = recovered_coef
                self.recovered_intercept[state] = recovered_intercept
                self.unit_coef[state] = unit_coef
        elif model == 'forward':
            for state in self.state_list:
                print(state, end = ':')
                active_g_indices, recovered_coef, _, recovered_intercept = forward_stepwise_regression(self.grouped_X[state], self.y[state], group_size, intercept_length, maxsteps = maxsteps)
                self.active_g_indices[state] = active_g_indices
                self.recovered_coef[state] = recovered_coef
                self.recovered_intercept[state] = recovered_intercept
        elif model == 'derandomKnock':
            self.states_curr_W = {}
            for state in self.state_list:
                if states_prev_W is None:
                    full_prev_W = None
                else:
                    full_prev_W = states_prev_W[state]
                print(state, end = ':')
                active_g_indices, recovered_coef, recovered_intercept, full_curr_W = derandomKnock(self.X[state], self.y[state], self.standardized_grouped_X[state], self.standardized_y[state], self.grouped_X[state], group_size, intercept_length, full_prev_W = full_prev_W, tau = tau, M = M, M_max = M_max, fdr = fdr, v = 1)
                self.active_g_indices[state] = active_g_indices
                self.recovered_coef[state] = recovered_coef
                self.recovered_intercept[state] = recovered_intercept
                self.states_curr_W[state] = full_curr_W
        else:
            raise ValueError("Invalid model provided. Use 'grpLasso', 'forward' or 'derandomKnock'.")

    def get_test(self, end_date: int, codes: List[str], testing_period: int):
        # X_test = self.X_test[primary_state]
        # y_test = self.y_test[primary_state]
        self.y_test = {}
        self.grouped_X_test = {}
        end_test = date_after(end_date, testing_period - 1)
        for state in self.state_list:
            X = self.state_data[state].loc[date_before(end_date, max(lag_set)):end_test][codes].to_numpy()
            y = self.target_data.loc[end_date:end_test, state].to_numpy()
            grouped_X = group_X(X, lag_set=lag_set)
            self.y_test[state] = y
            self.grouped_X_test[state] = grouped_X
        
### grpLasso
# SIS - grpLasso - KKT
# class celer.GroupLasso(groups=1, alpha=1.0, max_iter=100, max_epochs=50000, p0=10, verbose=0, tol=0.0001, prune=True, fit_intercept=True, weights=None, warm_start=False)
# (1 / (2 * n_samples)) * ||y - X w||^2_2 + alpha * \sum_g weights_g ||w_g||_2
def fit_grpLasso(standardized_grouped_X, standardized_y, grouped_X, y, X_std, y_std, group_size, intercept_length, alpha, weights = None):
    grpLasso_model = celer.GroupLasso(groups = group_size, alpha = alpha, tol = 1e-4, fit_intercept = False, weights = weights)
    grpLasso_model.fit(standardized_grouped_X, standardized_y)
    recovered_coef = recover_coefficients(grpLasso_model.coef_, np.repeat(X_std, group_size), y_std)
    recovered_intercept = recover_intercept(grouped_X, y, recovered_coef, intercept_length)
    return grpLasso_model, recovered_coef, recovered_intercept

def check_KKT(standardized_grouped_X, standardized_y, coef, g_indices, group_size, alpha, weights = None, tol = 1e-4):
    residual = standardized_y - np.dot(standardized_grouped_X, coef)
    check_stats = []
    for g_idx in range(0, standardized_grouped_X.shape[1], group_size):
        standardized_grouped_X_g = standardized_grouped_X[:, g_idx * group_size : g_idx * group_size + group_size]
        check_stat = np.linalg.norm(standardized_grouped_X_g.T.dot(residual), 2) / standardized_grouped_X_g.shape[0]
        check_stats.append(check_stat)
    check_stats = np.array(check_stats)
    if weights is None:
        weights = 1
    check_threshs = alpha * weights
    violated_g_indices = np.where(check_stats > check_threshs + tol)[0]
    violated_g_indices = list(set(violated_g_indices) - set(g_indices))
    return violated_g_indices

def SIS(X, y, nSel):
    correlations = np.abs(np.corrcoef(X.T, y)[:-1, -1])
    np.nan_to_num(correlations, copy=False)
    sorted_indices = np.argsort(correlations)[::-1]
    SIS_g_indices = sorted_indices[:nSel]
    return SIS_g_indices

# def SIS_grpLasso_KKT(X, y, standardized_grouped_X, standardized_y, grouped_X, X_std, y_std, group_size, intercept_length, alpha, weights = None):
#     SIS_g_indices = np.array([], dtype = int)
#     violated_g_indices = SIS(X, y, nSel = int(len(y) / group_size) - 1)
#     while len(violated_g_indices) > 0:
#         SIS_g_indices = np.append(SIS_g_indices, violated_g_indices)
#         SIS_v_indices = extend_indices(SIS_g_indices, group_size)
#         grpLasso_model, recovered_coef, recovered_intercept = fit_grpLasso(standardized_grouped_X[:, SIS_v_indices], standardized_y, grouped_X[:, SIS_v_indices], y, X_std[SIS_g_indices], y_std, group_size, intercept_length, alpha, weights)
#         full_unit_coef = np.zeros(standardized_grouped_X.shape[1])
#         full_unit_coef[SIS_v_indices] = grpLasso_model.coef_
#         violated_g_indices = check_KKT(standardized_grouped_X, standardized_y, full_unit_coef, group_size, alpha, weights)
#     full_recovered_coef = np.zeros(standardized_grouped_X.shape[1])
#     full_recovered_coef[SIS_v_indices] = recovered_coef
#     return SIS_g_indices, full_unit_coef, full_recovered_coef, recovered_intercept

def SIS_grpLasso_KKT(X, y, standardized_grouped_X, standardized_y, grouped_X, X_std, y_std, group_size, intercept_length, alpha, refit_prev = False, prev_g_indices = None, weights = None):
    if refit_prev:
        if prev_g_indices is None:
            violated_g_indices = SIS(X, y, nSel = int(len(y) / group_size) - 1)
        else:
            violated_g_indices = prev_g_indices
    else:
        violated_g_indices = SIS(X, y, nSel = int(len(y) / group_size) - 1)
    SIS_g_indices = np.array([], dtype = int)
    while len(violated_g_indices) > 0:
        SIS_g_indices = list(set(SIS_g_indices) | set(violated_g_indices))
        SIS_v_indices = extend_indices(SIS_g_indices, group_size)
        grpLasso_model, recovered_coef, recovered_intercept = fit_grpLasso(standardized_grouped_X[:, SIS_v_indices], standardized_y, grouped_X[:, SIS_v_indices], y, X_std[SIS_g_indices], y_std, group_size, intercept_length, alpha, weights)
        full_unit_coef = np.zeros(standardized_grouped_X.shape[1])
        full_unit_coef[SIS_v_indices] = grpLasso_model.coef_
        violated_g_indices = check_KKT(standardized_grouped_X, standardized_y, full_unit_coef, SIS_g_indices, group_size, alpha, weights)
    full_recovered_coef = np.zeros(standardized_grouped_X.shape[1])
    full_recovered_coef[SIS_v_indices] = recovered_coef
    active_v_indices = np.where(full_recovered_coef != 0)[0]
    active_g_indices = shrink_indices(active_v_indices, group_size)
    for g_idx in active_g_indices:
        print(all_codes[g_idx], end=' ')
    print()
    return active_g_indices, full_unit_coef, full_recovered_coef, recovered_intercept


### forward
def forward_stepwise_regression(grouped_X, y, group_size, intercept_length, maxsteps = 5):
    active_g_indices = []
    inactive_g_indices = list(range(int(grouped_X.shape[1] / group_size)))
    residual = y
    for step in range(maxsteps):
        best_g_idx = None
        best_corr = 0
        for g_idx in inactive_g_indices:
            corr = np.linalg.norm(np.corrcoef(grouped_X[:, g_idx * group_size : g_idx * group_size + group_size].T, residual)[:-1, -1], 2)
            if corr > best_corr:
                best_g_idx = g_idx
                best_corr = corr
        if best_g_idx is not None:
            active_g_indices.append(best_g_idx)
            inactive_g_indices.remove(best_g_idx)
            active_v_indices = extend_indices(active_g_indices, group_size)
            X_active = grouped_X[:, active_v_indices]
            # beta, _, _, _ = np.linalg.lstsq(X_active, y, rcond=None)
            extended_X_active = np.hstack([np.ones((grouped_X.shape[0], 1)), X_active])
            beta, _, _, _ = np.linalg.lstsq(extended_X_active, y, rcond=None)
            residual = y - extended_X_active @ beta
        else:
            break
    forward_coef = np.zeros(grouped_X.shape[1])
    forward_coef[active_v_indices] = beta[1:]
    forward_intercept = beta[0]
    recovered_intercept = recover_intercept(grouped_X, y, forward_coef, intercept_length)
    for g_idx in active_g_indices:
        # print(all_codes[g_idx], ':', np.mean(full_unit_coef[g_idx * group_size : g_idx * group_size + group_size]), end='\t')
        print(all_codes[g_idx], end=' ')
    print()
    return active_g_indices, forward_coef, forward_intercept, recovered_intercept

### derandomKnock
def derandomKnock_filter(standardized_grouped_X, standardized_y, group_size, prev_W = None, tau = 0.5, M = 10, M_max = 30, fdr = 0.1, v = 1):
    p = int(standardized_grouped_X.shape[1] / group_size)
    groups = np.repeat(np.arange(1, p + 1, 1), group_size)
    pi = np.zeros(p)
    curr_W = [] if prev_W is None else prev_W
    if len(curr_W) > M_max - M:
        curr_W = curr_W[-(M_max - M):]
    for m in range(M):
        kfilter = KnockoffFilter(ksampler = 'gaussian', fstat = 'lasso')
        kfilter.forward(standardized_grouped_X, standardized_y, groups = groups, fdr = fdr)
        W = kfilter.W
        curr_W.append(W)
    for W in curr_W:
        if np.sum(W < 0) < v:
            S = np.where(W > 0)[0] 
            pi[S] += 1
        else:
            order_w = np.argsort(np.abs(W))[::-1]
            sorted_w = W[order_w]
            negid = np.where(sorted_w < 0)[0]
            TT = negid[v - 1]
            S = np.where(sorted_w[:TT] > 0)[0]
            S = order_w[S]
            pi[S] += 1
    pi /= len(curr_W)
    S = np.where(pi >= min(tau, max(pi)))[0]
    return S, pi, curr_W

def derandomKnock_regression(grouped_X, y, S_g_indices):
    derandomKnock_model = LinearRegression()
    S_v_indices = extend_indices(S_g_indices, group_size)
    derandomKnock_model.fit(grouped_X[:, S_v_indices], y)
    derandomKnock_model_coef = np.zeros(grouped_X.shape[1])
    derandomKnock_model_coef[S_v_indices] = derandomKnock_model.coef_
    return derandomKnock_model_coef, derandomKnock_model.intercept_

def derandomKnock(X, y, standardized_grouped_X, standardized_y, grouped_X, group_size, intercept_length, full_prev_W = None, tau = 0.5, M = 10, M_max = 30, fdr = 0.1, v = 1):
    SIS_g_indices = SIS(X, y, nSel = int(len(y) / group_size) - 1)
    SIS_v_indices = extend_indices(SIS_g_indices, group_size)
    if full_prev_W is None:
        prev_W = None
    else:
        prev_W = [W[SIS_g_indices] for W in full_prev_W]
    S, pi, curr_W = derandomKnock_filter(standardized_grouped_X[:, SIS_v_indices], standardized_y, group_size, prev_W, tau, M, M_max, fdr, v)
    S_g_indices = SIS_g_indices[S]
    for ss in S:
        print(all_codes[SIS_g_indices[ss]], ':', pi[ss], end=' ')
    print()
    full_curr_W = [np.zeros(X.shape[1]) for w_idx in range(len(curr_W))]
    for w_idx, W in enumerate(curr_W):
        full_curr_W[w_idx][SIS_g_indices] = W
    recovered_coef, _ = derandomKnock_regression(grouped_X, y, S_g_indices)
    recovered_intercept = recover_intercept(grouped_X, y, recovered_coef, intercept_length)
    return S_g_indices, recovered_coef, recovered_intercept, full_curr_W

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    lag_set = np.array([14, 7, 0])
    group_size = len(lag_set)
    state_list = ['VT', 'MA', 'ME', 'NH', 'CT', 'RI']
    
    training_period = 150
    testing_period = 30
    step_size = 30
    intercept_length = 60
    start_date = 20200801
    end_date = date_after(start_date, training_period - 1)
    start_date_list = []
    end_date_list = []
    while end_date < 20230801:
        start_date_list.append(start_date)
        end_date_list.append(end_date)
        start_date = date_after(start_date, step_size)
        end_date = date_after(start_date, training_period - 1)
    
    target_data = pd.read_csv('data/HHShospitalized.txt', index_col = 0)
    target_data = target_data.loc[:, state_list].astype(float).apply(moving_average_smoother).iloc[6:]
    state_data_dict = {state: pd.read_csv(f'data/{state}phecodes.csv', index_col=0, header=0).astype(float).apply(moving_average_smoother).iloc[6:] for state in state_list}
    all_codes = state_data_dict[state_list[0]].columns.to_numpy()
    
    data_handler = HealthDataHandler(state_list, target_data, state_data_dict)
    # states_prev_W = None
    # for d_idx in range(len(start_date_list)):
    #     data_handler.get_X_y(start_date = start_date_list[d_idx], end_date = end_date_list[d_idx], codes = all_codes)
    #     print(start_date_list[d_idx], '-', end_date_list[d_idx])
    #     data_handler.run_all_single('derandomKnock', group_size, intercept_length, alpha = 0.3, maxsteps = 5, states_prev_W = states_prev_W, tau = 0.5, M = 10, M_max = 30, fdr = 0.1)
    #     with open('run_all_single'+str(end_date_list[d_idx])+'_120.pkl', 'wb') as file:
    #         pickle.dump(data_handler, file)
    #     states_prev_W = data_handler.states_curr_W
    
    # run_all_single(self, model, group_size, intercept_length, alpha = 0.3, refit_prev = False, states_prev_g = None, maxsteps = 5, states_prev_W = None, tau = 0.5, M = 10, M_max = 30, fdr = 0.1):
    refit_prev = True
    states_prev_g = None
    y_dict = {}
    pred_dict = {}
    
    start_plot = datetime.strptime(str(end_date_list[0]), '%Y%m%d')
    end_plot = datetime.strptime(str(date_after(end_date_list[-1], testing_period - 1)), '%Y%m%d')
    date_plot_list = []
    while start_plot <= end_plot:
        date_plot_list.append(start_plot)
        start_plot += timedelta(days=1)
    
    for state in state_list:
        y_dict[state] = []
        pred_dict[state] = []
    for d_idx in range(len(start_date_list)):
        data_handler.get_X_y(start_date = start_date_list[d_idx], end_date = end_date_list[d_idx], codes = all_codes)
        data_handler.get_test(end_date = end_date_list[d_idx], codes = all_codes, testing_period = testing_period)
        print(start_date_list[d_idx], '-', end_date_list[d_idx])
        data_handler.run_all_single('grpLasso', group_size, intercept_length, alpha = 0.4, refit_prev = refit_prev, states_prev_g = states_prev_g, maxsteps = 5, states_prev_W = None, tau = 0.5, M = 10, M_max = 30, fdr = 0.1)
        states_prev_g = data_handler.active_g_indices
        for state in state_list:
            y_dict[state].append(data_handler.y_test[state])
            pred_dict[state].append(np.dot(data_handler.grouped_X_test[state], data_handler.recovered_coef[state]) + data_handler.recovered_intercept[state])
    
    for state in state_list:
        plt.figure(figsize=(12, 8))
        plt.plot(np.array(date_plot_list), np.array(y_dict[state]).flatten(), label = 'y')
        plt.plot(np.array(date_plot_list), np.array(pred_dict[state]).flatten(), label = 'pred')
        plt.title(state)
        plt.xticks(rotation = 30)
        plt.legend()
    