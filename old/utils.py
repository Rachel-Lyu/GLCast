import numpy as np
import pandas as pd
import pickle
import celer
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from knockpy import KnockoffFilter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

state_code = {'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08', 'CT': '09', 'DE': '10', 'DC': '11', 'FL': '12', 'GA': '13', 
              'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24', 'MA': '25', 
              'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34', 'NM': '35', 'NY': '36', 
              'NC': '37', 'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', # 'ND': '38',
              'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55', 'WY': '56', 'PR': '72'}

state_abbr = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA',
              'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA',
              'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY',
              'NC', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', # 'ND', 
              'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

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
                active_g_indices, unit_coef, recovered_coef, recovered_intercept = SIS_grpLasso_KKT(self.X[state], self.y[state], self.standardized_grouped_X[state], self.standardized_y[state], self.grouped_X[state], self.X_std[state], self.y_std[state], group_size, intercept_length, alpha = alpha, refit_prev = refit_prev, prev_g_indices = prev_g_indices)
                # model_coef = unit_coef if coef_method == 'standardized' else recovered_coef
                self.active_g_indices[state] = active_g_indices
                self.recovered_coef[state] = recovered_coef
                self.recovered_intercept[state] = recovered_intercept
                self.unit_coef[state] = unit_coef
        elif model == 'forward':
            for state in self.state_list:
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
    
    def get_informative_states(self, primary_state, state_method, auxiliary_state_list = None, nStates = 5):
        # marginal_diff
        if state_method == 'marginal_diff':
            if auxiliary_state_list is None:
                auxiliary_state_list = [s for s in self.state_list if s != primary_state]
            if nStates is not None:
                self.informative_state_list = auxiliary_state_list[self.sort_marginal_diff(primary_state, auxiliary_state_list)[:nStates]]
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
    
    def Q_aggregation(self, B, X_test, y_test, total_step = 10):
        # theta.hat<- exp(-colSums((y.test-X.test%*%B)^2)/2)
        theta_hat = np.exp(-np.sum((y_test - np.dot(X_test, B)) ** 2, axis = 0) / 2)
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
            # theta.hat<- exp(-colSums((y.test-X.test%*%B)^2)/2+colSums((as.vector(X.test%*%beta)-X.test%*%B)^2)/8)
            theta_hat = np.exp(-np.sum((y_test - np.dot(X_test, B)) ** 2, axis = 0) / 2 + 
                            np.sum((np.dot(X_test, beta).reshape(-1, 1) - np.dot(X_test, B)) ** 2, axis = 0) / 8)
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
    
    def run_all_agg(primary_state, state_method, auxiliary_state_list = None, nStates = 5, total_step = 10):
        self.recovered_coef = {}
        self.recovered_intercept = {}
        for primary_state in self.state_list:
            self.get_informative_states(primary_state, state_method, auxiliary_state_list, nStates)
            B = np.zeros((len(self.informative_state_list) + 1, self.nV))
            B[0] = self.recovered_coef[primary_state]
            for s_idx, informative_state in enumerate(self.informative_state_list):
                B[s_idx + 1] = self.recovered_coef[informative_state]
            B = B.T
            theta_hat, beta = self.Q_aggregation(B, primary_state, self.informative_state_list, total_step)
    
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
