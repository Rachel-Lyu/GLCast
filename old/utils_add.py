import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import pickle
import celer
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from knockpy import KnockoffFilter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

state_abbr = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA',
              'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA',
              'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY',
              'NC', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', # 'ND', 
              'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'] # , 'PR'

# mega_list: list of lists of sibling geo_key_ids
mega_list = [['VT', 'MA', 'ME', 'NH', 'CT', 'RI'],
             ['OR', 'WA', 'AK', 'ID'],
             ['NJ', 'NY'], # 'PR', 
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
    def __init__(self, state_list: List[str], lag_set: np.ndarray, report_freq_days: int, target_data: pd.DataFrame, state_data: Dict[str, pd.DataFrame]):
        self.state_list = state_list
        self.lag_set = lag_set
        self.report_freq_days = report_freq_days
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
            X = self.state_data[state].loc[date_before(start_date, max(self.lag_set) * self.report_freq_days):end_date - self.report_freq_days][codes].to_numpy()
            y = self.target_data.loc[start_date:end_date - self.report_freq_days, state].to_numpy()
            standardized_X, X_mean, X_std = standardize_X(X)
            standardized_y, y_mean, y_std = standardize_y(y)
            standardized_grouped_X = group_X(standardized_X, lag_set = self.lag_set)
            grouped_X = group_X(X, lag_set = self.lag_set)
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

    def run_all_single(self, intercept_length, model, alpha = 0.3, strong_g_indices = None, states_prev_g = None, maxsteps = 5, states_prev_W = None, tau = 0.5, M = 10, M_max = 30, fdr = 0.1):
        # grpLasso derandomKnock forward
        group_size = len(self.lag_set)
        self.active_g_indices = {}
        self.recovered_coef = {}
        self.recovered_intercept = {}
        # if strong_codes is not None:
        #     strong_g_indices = np.where(np.isin(all_codes, strong_codes))[0]
        #     strong_v_indices = extend_indices(strong_g_indices, group_size)
        if model == 'grpLasso':
            self.unit_coef = {}
            for state in self.state_list:
                if states_prev_g is None:
                    prev_g_indices = None
                else:
                    prev_g_indices = states_prev_g[state]
                active_g_indices, unit_coef, recovered_coef, recovered_intercept = SIS_grpLasso_KKT(self.X[state], self.y[state], self.standardized_grouped_X[state], self.standardized_y[state], self.grouped_X[state], self.X_std[state], self.y_std[state], group_size, intercept_length, alpha = alpha, strong_g_indices = strong_g_indices, prev_g_indices = prev_g_indices)
                # model_coef = unit_coef if coef_method == 'standardized' else recovered_coef
                self.active_g_indices[state] = active_g_indices
                self.recovered_coef[state] = recovered_coef
                self.recovered_intercept[state] = recovered_intercept
                self.unit_coef[state] = unit_coef
        elif model == 'forward':
            for state in self.state_list:
                active_g_indices, recovered_coef, recovered_intercept = forward_stepwise_regression(self.grouped_X[state], self.y[state], group_size, intercept_length, strong_g_indices = strong_g_indices, maxsteps = maxsteps)
                self.active_g_indices[state] = active_g_indices
                self.recovered_coef[state] = recovered_coef
                self.recovered_intercept[state] = recovered_intercept
        elif model == 'forward_backward':
            for state in self.state_list:
                if states_prev_g is None:
                    prev_g_indices = None
                else:
                    prev_g_indices = states_prev_g[state]
                active_g_indices, recovered_coef, recovered_intercept = forward_backward_stepwise_regression(self.grouped_X[state], self.y[state], group_size, intercept_length, strong_g_indices = strong_g_indices, maxsteps = maxsteps, prev_g_indices = prev_g_indices)
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
                active_g_indices, recovered_coef, recovered_intercept, full_curr_W = derandomKnock(self.X[state], self.y[state], self.standardized_grouped_X[state], self.standardized_y[state], self.grouped_X[state], group_size, intercept_length, strong_g_indices = strong_g_indices, full_prev_W = full_prev_W, tau = tau, M = M, M_max = M_max, fdr = fdr, v = 1)
                self.active_g_indices[state] = active_g_indices
                self.recovered_coef[state] = recovered_coef
                self.recovered_intercept[state] = recovered_intercept
                self.states_curr_W[state] = full_curr_W
        else:
            raise ValueError("Invalid model provided. Use 'grpLasso', 'forward' or 'derandomKnock'.")

    def get_test(self, start_test: int, codes: List[str], period_length: int):
        self.y_test = {}
        self.grouped_X_test = {}
        end_test = date_after(start_test, period_length)
        for state in self.state_list:
            X = self.state_data[state].loc[date_before(start_test, max(self.lag_set) * self.report_freq_days):end_test - self.report_freq_days][codes].to_numpy()
            y = self.target_data.loc[start_test:end_test - self.report_freq_days, state].to_numpy()
            grouped_X = group_X(X, lag_set = self.lag_set)
            self.y_test[state] = y
            self.grouped_X_test[state] = grouped_X
        self.test_dates = self.target_data.loc[start_test:end_test - self.report_freq_days, state].index.to_numpy()

    def get_informative_states(self, primary_state, state_method, auxiliary_state_list = None, nStates = 5):
        # marginal_diff
        if state_method == 'marginal_diff':
            if auxiliary_state_list is None:
                auxiliary_state_list = [s for s in self.state_list if s != primary_state]
            if nStates is not None:
                self.informative_state_list = np.array(auxiliary_state_list)[self.sort_marginal_diff(primary_state, auxiliary_state_list)[:nStates]]
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
    
    def Q_aggregation(self, B, X_test, y_test, total_step = 10, selection = False):
        # y_test -= y_test.mean()
        y_test = y_test.reshape(-1, 1)
        XB = np.dot(X_test, B)
        # XB -= XB.mean(axis = 0)
        # if(selection){#select beta.hat with smallest prediction error
        if not selection: 
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
        if selection or np.isnan(theta_hat).any():
            # khat<-which.min(colSums((y.test-X.test%*%B)^2))
            # theta.hat<-rep(0, ncol(B))
            # theta.hat[khat] <- 1
            # beta=B[,khat]
            khat = np.argmin(np.mean((y_test - XB) ** 2, axis = 0))
            theta_hat = np.zeros(B.shape[1])
            theta_hat[khat] = 1
            beta = B[:, khat]
        return theta_hat, beta
    
    def run_all_agg(self, intercept_length, state_method, auxiliary_state_list = None, nStates = 5, total_step = 10, selection = False):
        group_size = len(self.lag_set)
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

def SIS_grpLasso_KKT(X, y, standardized_grouped_X, standardized_y, grouped_X, X_std, y_std, group_size, intercept_length, alpha, strong_g_indices = None, prev_g_indices = None, weights = None):
    if prev_g_indices is None or len(prev_g_indices) == 0:
        violated_g_indices = SIS(X, y, nSel = int(len(y) / group_size) - 1)
    else:
        violated_g_indices = prev_g_indices
    if strong_g_indices is None:
        strong_g_indices = []
    SIS_g_indices = np.array([], dtype = int)
    while len(violated_g_indices) > 0:
        SIS_g_indices = list(set(SIS_g_indices) | set(violated_g_indices) | set(strong_g_indices))
        SIS_v_indices = extend_indices(SIS_g_indices, group_size)
        grpLasso_model, recovered_coef, recovered_intercept = fit_grpLasso(standardized_grouped_X[:, SIS_v_indices], standardized_y, grouped_X[:, SIS_v_indices], y, X_std[SIS_g_indices], y_std, group_size, intercept_length, alpha, weights)
        full_unit_coef = np.zeros(standardized_grouped_X.shape[1])
        full_unit_coef[SIS_v_indices] = grpLasso_model.coef_
        violated_g_indices = check_KKT(standardized_grouped_X, standardized_y, full_unit_coef, SIS_g_indices, group_size, alpha, weights)
    full_recovered_coef = np.zeros(standardized_grouped_X.shape[1])
    full_recovered_coef[SIS_v_indices] = recovered_coef
    active_v_indices = np.where(full_recovered_coef != 0)[0]
    active_g_indices = shrink_indices(active_v_indices, group_size)
    return active_g_indices, full_unit_coef, full_recovered_coef, recovered_intercept


### forward
def forward_stepwise_regression(grouped_X, y, group_size, intercept_length, strong_g_indices = None, maxsteps = 5):
    active_g_indices = []
    inactive_g_indices = list(range(int(grouped_X.shape[1] / group_size)))
    if strong_g_indices is not None:
        for g_idx in strong_g_indices:
            active_g_indices.append(g_idx)
            inactive_g_indices.remove(g_idx)
        active_v_indices = extend_indices(active_g_indices, group_size)
        X_active = grouped_X[:, active_v_indices]
        # beta, _, _, _ = np.linalg.lstsq(X_active, y, rcond=None)
        extended_X_active = np.hstack([np.ones((grouped_X.shape[0], 1)), X_active])
        beta, _, _, _ = np.linalg.lstsq(extended_X_active, y, rcond=None)
        residual = y - extended_X_active @ beta
    else:
        residual = y
        beta = None
    # for step in range(maxsteps):
    while len(active_g_indices) < maxsteps:
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
    if beta is not None:
        forward_coef[active_v_indices] = beta[1:]
    recovered_intercept = recover_intercept(grouped_X, y, forward_coef, intercept_length)
    return active_g_indices, forward_coef, recovered_intercept

### forward_backward
def forward_backward_stepwise_regression(grouped_X, y, group_size, intercept_length, strong_g_indices = None, maxsteps = 5, prev_g_indices = None):
    active_g_indices = []
    inactive_g_indices = list(range(int(grouped_X.shape[1] / group_size)))
    if strong_g_indices is None:
        strong_g_indices = []
    if prev_g_indices is None:
        prev_g_indices = strong_g_indices
    else:
        prev_g_indices = list(set(prev_g_indices) | set(strong_g_indices))
    if prev_g_indices is None or len(prev_g_indices) == 0:
        residual = y
        beta = None
    else:
        for g_idx in prev_g_indices:
            active_g_indices.append(g_idx)
            inactive_g_indices.remove(g_idx)
        active_v_indices = extend_indices(active_g_indices, group_size)
        X_active = grouped_X[:, active_v_indices]
        extended_X_active = np.hstack([np.ones((grouped_X.shape[0], 1)), X_active])
        beta, _, _, _ = np.linalg.lstsq(extended_X_active, y, rcond=None)
        residual = y - extended_X_active @ beta

    # for step in range(maxsteps):
    while len(active_g_indices) < 2 * maxsteps:
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

    while len(active_g_indices) > maxsteps:
        worst_g_idx = None
        smallest_improvement = np.inf
        for g_idx in active_g_indices:
            if g_idx not in strong_g_indices:
                temp_active_g_indices = active_g_indices.copy()
                temp_active_g_indices.remove(g_idx)
                temp_active_v_indices = extend_indices(temp_active_g_indices, group_size)
                X_active_temp = grouped_X[:, temp_active_v_indices]
                extended_X_active_temp = np.hstack([np.ones((grouped_X.shape[0], 1)), X_active_temp])
                beta_temp, _, _, _ = np.linalg.lstsq(extended_X_active_temp, y, rcond=None)
                residual_temp = y - extended_X_active_temp @ beta_temp
                improvement = np.sum(residual_temp**2) - np.sum(residual**2)
                if improvement < smallest_improvement:
                    smallest_improvement = improvement
                    worst_g_idx = g_idx
        active_g_indices.remove(worst_g_idx)
        active_v_indices = extend_indices(active_g_indices, group_size)
        X_active = grouped_X[:, active_v_indices]
        extended_X_active = np.hstack([np.ones((grouped_X.shape[0], 1)), X_active])
        beta, _, _, _ = np.linalg.lstsq(extended_X_active, y, rcond=None)
        residual = y - extended_X_active @ beta
    
    stepwise_coef = np.zeros(grouped_X.shape[1])
    if beta is not None:
        stepwise_coef[active_v_indices] = beta[1:]
    recovered_intercept = recover_intercept(grouped_X, y, stepwise_coef, intercept_length)
    return active_g_indices, stepwise_coef, recovered_intercept

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

def derandomKnock_regression(grouped_X, y, S_g_indices, group_size):
    derandomKnock_model = LinearRegression()
    S_v_indices = extend_indices(S_g_indices, group_size)
    derandomKnock_model.fit(grouped_X[:, S_v_indices], y)
    derandomKnock_model_coef = np.zeros(grouped_X.shape[1])
    derandomKnock_model_coef[S_v_indices] = derandomKnock_model.coef_
    return derandomKnock_model_coef, derandomKnock_model.intercept_

def derandomKnock(X, y, standardized_grouped_X, standardized_y, grouped_X, group_size, intercept_length, strong_g_indices = None, full_prev_W = None, tau = 0.5, M = 10, M_max = 30, fdr = 0.1, v = 1):
    SIS_g_indices = SIS(X, y, nSel = int(len(y) / group_size) - 1)
    if np.unique(standardized_y).shape[0] > 2:
        SIS_v_indices = extend_indices(SIS_g_indices, group_size)
        if full_prev_W is None:
            prev_W = None
        else:
            prev_W = [W[SIS_g_indices] for W in full_prev_W]
        S, pi, curr_W = derandomKnock_filter(standardized_grouped_X[:, SIS_v_indices], standardized_y, group_size, prev_W, tau, M, M_max, fdr, v)
        S = list(set(S) | set(strong_g_indices))
        S_g_indices = SIS_g_indices[S]
        full_curr_W = [np.zeros(X.shape[1]) for w_idx in range(len(curr_W))]
        for w_idx, W in enumerate(curr_W):
            full_curr_W[w_idx][SIS_g_indices] = W
    else:
        S_g_indices = SIS_g_indices
        full_curr_W = full_prev_W
    recovered_coef, _ = derandomKnock_regression(grouped_X, y, S_g_indices, group_size)
    recovered_intercept = recover_intercept(grouped_X, y, recovered_coef, intercept_length)
    return S_g_indices, recovered_coef, recovered_intercept, full_curr_W