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

# mega_list: list of lists of sibling geo_key_ids
mega_list = [['VT', 'MA', 'ME', 'NH', 'CT', 'RI'],
             ['OR', 'WA', 'AK', 'ID'],
             ['PR', 'NJ', 'NY'],
             ['WV', 'MD', 'PA', 'DC', 'DE', 'VA'],
             ['KY', 'MS', 'NC', 'TN', 'AL', 'SC', 'FL', 'GA'],
             ['IL', 'IN', 'WI', 'MI', 'MN', 'OH'],
             ['LA', 'NM', 'AR', 'OK', 'TX'],
             ['NE', 'KS', 'IA', 'MO'],
             ['WY', 'MT', 'CO', 'SD', 'UT'], # 'ND', 
             ['HI', 'NV', 'AZ', 'CA']]

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
        self.X_mean = {}
        self.X_std = {}
        self.y = {}
        self.standardized_y = {}
        self.y_mean = {}
        self.y_std = {}
        self.standardized_grouped_X = {}
        self.grouped_X = {}
        for state in self.state_list:
            X = self.state_data[state].loc[date_before(start_date, max(lag_set)):end_date][codes].to_numpy()
            y = self.target_data.loc[start_date:end_date, state].to_numpy()
            standardized_X, X_mean, X_std = standardize_X(X)
            standardized_y, y_mean, y_std = standardize_y(y)
            standardized_grouped_X = group_X(standardized_X, lag_set=lag_set)
            grouped_X = group_X(X, lag_set=lag_set)
            X = X[-(len(y)):]
            standardized_X = standardized_X[-(len(y)):]
            self.X[state] = X
            self.standardized_X[state] = standardized_X
            self.X_mean[state] = X_mean
            self.X_std[state] = X_std
            self.y[state] = y
            self.standardized_y[state] = standardized_y
            self.y_mean[state] = y_mean
            self.y_std[state] = y_std
            self.standardized_grouped_X[state] = standardized_grouped_X
            self.grouped_X[state] = grouped_X
        self.nS = len(y)
        self.nG = X.shape[1]
        self.nV = grouped_X.shape[1]

    def run_all_single(self, model, alpha = 0.3, refit_prev = False, states_prev_g = None, maxsteps = 5, states_prev_W = None, tau = 0.5, M = 10, M_max = 30, fdr = 0.1):
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

    def get_test(self, start_test: int, codes: List[str], period_length: int):
        self.y_test = {}
        self.grouped_X_test = {}
        end_test = date_after(start_test, period_length - 1)
        for state in self.state_list:
            X = self.state_data[state].loc[date_before(start_test, max(lag_set)):end_test][codes].to_numpy()
            y = self.target_data.loc[start_test:end_test, state].to_numpy()
            grouped_X = group_X(X, lag_set=lag_set)
            self.y_test[state] = y
            self.grouped_X_test[state] = grouped_X

    def get_informative_states(self, primary_state, state_method, auxiliary_state_list = None, nStates = 5):
        # marginal_diff
        if state_method == 'marginal_diff':
            if auxiliary_state_list is None:
                auxiliary_state_list = [s for s in self.state_list if s != primary_state]
            if nStates is not None:
                self.informative_state_list = np.array(auxiliary_state_list)[self.sort_marginal_diff(primary_state, auxiliary_state_list)[:nStates]]
                print(primary_state, self.informative_state_list)
                return self.informative_state_list
            else:
                raise ValueError("nStates must not be None for 'marginal_diff' method.")
        # sibling
        elif state_method == 'sibling':
            for sublist in mega_list:
                if primary_state in sublist:
                    self.informative_state_list = [s for s in sublist if s != primary_state]
                    return self.informative_state_list
            raise ValueError(f"{primary_state} not found in any sibling groups.")
        else:
            raise ValueError("Invalid state_method provided. Use 'marginal_diff' or 'sibling'.")
    
    def sort_marginal_diff(self, primary_state, auxiliary_state_list):
        primary_Xty = np.dot(self.standardized_X[primary_state].T, self.standardized_y[primary_state]) / self.nS
        auxiliary_state_Xty = np.array([np.dot(self.standardized_X[auxiliary_state].T, self.standardized_y[auxiliary_state]) / self.nS for auxiliary_state in auxiliary_state_list])
        abs_diff = np.abs(auxiliary_state_Xty - primary_Xty)
        Rhat = []
        for s_idx, auxiliary_state in enumerate(auxiliary_state_list):
            abs_diff_s = abs_diff[s_idx]
            screened_largest_diff = abs_diff_s[np.argsort(abs_diff_s)[-int(self.nS/3):]]
            Rhat.append(np.sum(screened_largest_diff ** 2))
        return np.argsort(Rhat)
    
        # X_test = self.X_test[primary_state]
        # y_test = self.y_test[primary_state]
    
    def Q_aggregation(self, B, X_test, y_test, total_step = 10, selection = False):
        # y_test -= y_test.mean()
        y_test = y_test.reshape(-1, 1)
        XB = np.dot(X_test, B)
        # XB -= XB.mean(axis = 0)
        # if(selection){#select beta.hat with smallest prediction error
        # khat<-which.min(colSums((y.test-X.test%*%B)^2))
        # theta.hat<-rep(0, ncol(B))
        # theta.hat[khat] <- 1
        # beta=B[,khat]
        if selection: 
            khat = np.argmin(np.mean((y_test - XB) ** 2, axis = 0))
            theta_hat = np.zeros(B.shape[1])
            theta_hat[khat] = 1
            beta = B[:, khat]
        else: 
        # }else{#Q-aggregation
            # theta.hat<- exp(-colSums((y.test-X.test%*%B)^2)/2)
            # theta_hat = np.exp(-np.sum((y_test - np.dot(X_test, B)) ** 2, axis = 0) / 2)
            theta_hat = np.exp(-np.mean((y_test - XB) ** 2, axis = 0) / 2)
            # theta.hat=theta.hat/sum(theta.hat)
            theta_hat /= np.sum(theta_hat)
            # theta.old=theta.hat
            theta_old = theta_hat.copy()
            # beta<-as.numeric(B%*%theta.hat)
            beta = np.dot(B, theta_hat)
            # beta.ew<-beta
            beta_ew = beta.copy()
            # for(ss in 1:total.step){
            for ss in range(total_step):
                # print(np.round(theta_hat, 2))
                Xbeta = np.dot(X_test, beta)
                # Xbeta -= Xbeta.mean()
                # theta.hat<- exp(-colSums((y.test-X.test%*%B)^2)/2+colSums((as.vector(X.test%*%beta)-X.test%*%B)^2)/8)
                theta_hat = np.exp(-np.mean((y_test - XB) ** 2, axis = 0) / 2 + 
                                np.mean((Xbeta.reshape(-1, 1) - XB) ** 2, axis = 0) / 8)
                # theta.hat<-theta.hat/sum(theta.hat)
                theta_hat /= np.sum(theta_hat)
                # beta<- as.numeric(B%*%theta.hat*1/4+3/4*beta)
                beta = np.dot(B, theta_hat) * 1/4 + 3/4 * beta
                # if(sum(abs(theta.hat-theta.old))<10^(-3)){break}
                if np.sum(np.abs(theta_hat - theta_old)) < 1e-3:
                    break
                # theta.old=theta.hat
                theta_old = theta_hat.copy()
            # }
        return theta_hat, beta
    
    def run_all_agg(self, state_method, auxiliary_state_list = None, nStates = 5, total_step = 10, selection = False):
        self.recovered_coef_agg = {}
        self.recovered_intercept_agg = {}
        for primary_state in self.state_list:
            self.get_informative_states(primary_state, state_method, auxiliary_state_list, nStates)
            B = np.zeros((len(self.informative_state_list) + 1, self.nV))
            B[0] = self.recovered_coef[primary_state] / self.y_std[primary_state] * np.repeat(self.X_std[primary_state], group_size)
            for s_idx, informative_state in enumerate(self.informative_state_list):
                B[s_idx + 1] = self.recovered_coef[informative_state] / self.y_std[primary_state] * np.repeat(self.X_std[primary_state], group_size)
            B = B.T
            X_test = (self.grouped_X_test[primary_state] - np.repeat(self.X_mean[primary_state], group_size)) / np.repeat(self.X_std[primary_state], group_size)
            y_test = (self.y_test[primary_state] - self.y_mean[primary_state]) / self.y_std[primary_state]
            theta_hat, beta = self.Q_aggregation(B, X_test, y_test, total_step, selection)
            print(np.round(theta_hat, 2))
            self.recovered_coef_agg[primary_state] = recover_coefficients(beta, np.repeat(self.X_std[primary_state], group_size), self.y_std[primary_state])
            self.recovered_intercept[primary_state] = recover_intercept(self.grouped_X_test[primary_state], self.y_test[primary_state], self.recovered_coef[primary_state], intercept_length)
            self.recovered_intercept_agg[primary_state] = recover_intercept(self.grouped_X_test[primary_state], self.y_test[primary_state], self.recovered_coef_agg[primary_state], intercept_length)
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
        if prev_g_indices is None or len(prev_g_indices) == 0:
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
    state_list = ['MA', 'CT', 'RI', 'NJ', 'NY', 'PA', 'VA']
    
    train_period = 120
    validation_period = 60
    intercept_length = 30
    test_period = 180
    step_size = 30
    
    train_start_date_list = []
    validation_start_date_list = []
    test_start_date_list = []
    ### train_start_date = 20200801
    train_start_date = 20211201
    validation_start_date = date_after(train_start_date, train_period - 1)
    test_start_date = date_after(validation_start_date, validation_period - 1)
    while test_start_date < 20230201:
        train_start_date_list.append(train_start_date)
        validation_start_date_list.append(validation_start_date)
        test_start_date_list.append(test_start_date)
        train_start_date = date_after(train_start_date, step_size)
        validation_start_date = date_after(train_start_date, train_period - 1)
        test_start_date = date_after(validation_start_date, validation_period - 1)
    
    ### target_data = pd.read_csv('data/HHShospitalized.txt', index_col = 0)
    target_data = pd.read_csv('data/HHSflu.txt', index_col = 0)
    target_data = target_data.loc[:, state_list].astype(float).apply(moving_average_smoother).iloc[6:]
    state_data_dict = {state: pd.read_csv(f'data/{state}phecodes.csv', index_col=0, header=0).astype(float).apply(moving_average_smoother).iloc[6:] for state in state_list}
    all_codes = state_data_dict[state_list[0]].columns.to_numpy()
    
    data_handler = HealthDataHandler(state_list, target_data, state_data_dict)
    y_dict = {}
    pred_dict = {}
    pred_agg_dict = {}
    date_plot_list = []
    for state in state_list:
        y_dict[state] = []
        pred_dict[state] = []
        pred_agg_dict[state] = []
    
    states_prev_W = None
    refit_prev = True
    states_prev_g = None
    selection = False
    for d_idx in range(len(train_start_date_list)):
        start_plot = datetime.strptime(str(test_start_date_list[d_idx]), '%Y%m%d')
        end_plot = datetime.strptime(str(date_after(test_start_date_list[d_idx], test_period - 1)), '%Y%m%d')
        date_plot = pd.date_range(start=start_plot, end=end_plot).tolist()
        date_plot_list.append(date_plot)
        
        data_handler.get_X_y(start_date = train_start_date_list[d_idx], end_date = validation_start_date_list[d_idx], codes = all_codes)
        data_handler.get_test(start_test = validation_start_date_list[d_idx], codes = all_codes, period_length = validation_period)
        print(train_start_date_list[d_idx], '-', test_start_date_list[d_idx])
        
        data_handler.run_all_single('grpLasso', alpha = 0.3, refit_prev = refit_prev, states_prev_g = states_prev_g, maxsteps = 5, states_prev_W = states_prev_W, tau = 0.5, M = 10, M_max = 30, fdr = 0.1)
        data_handler.run_all_agg(state_method = 'marginal_diff', auxiliary_state_list = None, nStates = 5, total_step = 10, selection = selection)
        # states_prev_W = data_handler.states_curr_W
        states_prev_g = data_handler.active_g_indices
        # with open(filename, 'wb') as file:
        #     pickle.dump(data_handler, file)
        
        data_handler.get_test(start_test = test_start_date_list[d_idx], codes = all_codes, period_length = test_period)
        for state in state_list:
            y_dict[state].append(data_handler.y_test[state])
            pred_dict[state].append(np.dot(data_handler.grouped_X_test[state], data_handler.recovered_coef[state]) + data_handler.recovered_intercept[state])
            pred_agg_dict[state].append(np.dot(data_handler.grouped_X_test[state], data_handler.recovered_coef_agg[state]) + data_handler.recovered_intercept_agg[state])

    for state in state_list:
        plt.figure(figsize=(10, 6))
        for d_idx in range(len(train_start_date_list)):
            plt.plot(date_plot_list[d_idx], y_dict[state][d_idx], c = 'k', lw = 2)
            plt.plot(date_plot_list[d_idx], pred_dict[state][d_idx], alpha = 0.8, lw = 2)
        plt.title(state + ', single')
        plt.xticks(rotation = 30)
        plt.figure(figsize=(10, 6))
        for d_idx in range(len(train_start_date_list)):
            plt.plot(date_plot_list[d_idx], y_dict[state][d_idx], c = 'k', lw = 2)
            plt.plot(date_plot_list[d_idx], pred_agg_dict[state][d_idx], alpha = 0.8, lw = 2)
        plt.title(state + ', aggregated')
        plt.xticks(rotation = 30)
    # data_handler.get_informative_states('PA', state_method = 'sibling', auxiliary_state_list = None, nStates = 3)
    # data_handler.get_informative_states('PA', state_method = 'marginal_diff', auxiliary_state_list = None, nStates = 3)