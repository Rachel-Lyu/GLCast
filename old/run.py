import pandas as pd
import numpy as np
# from utils import *
lag_set = np.array([14, 7, 0])
group_size = len(lag_set)
intercept_length = 7

state_list = ['VT', 'MA', 'ME', 'NH', 'CT', 'RI']
target_data = pd.read_csv('data/HHShospitalized.txt', index_col = 0)
target_data = target_data.loc[:, state_list].astype(float).apply(moving_average_smoother).iloc[6:]
state_data_dict = {state: pd.read_csv(f'data/{state}phecodes.csv', index_col=0, header=0).astype(float).apply(moving_average_smoother).iloc[6:] for state in state_list}

data_handler = HealthDataHandler(state_list, target_data, state_data_dict)
# state_list: List[str], target_data: pd.DataFrame, state_data: Dict[str, pd.DataFrame]

all_codes = state_data_dict[state_list[0]].columns.to_numpy()

# TEST
state = 'MA'

data_handler.get_X_y(start_date = 20200801, end_date = 20210801, codes = all_codes)
data_handler.run_all_single('derandomKnock', group_size, intercept_length, alpha = 0.3, maxsteps = 5, prev_W = None, tau = 0.2, M = 5, M_max = 30, fdr = 0.1)
active_g_indices, forward_coef, forward_intercept = forward_stepwise_regression(data_handler.grouped_X[state], data_handler.y[state], group_size, maxsteps = 5)

S, pi, curr_W = derandomKnock(data_handler.standardized_grouped_X[state], data_handler.standardized_y[state], prev_W = None, group_size = group_size, tau = 0.5, M = 2, M_max = 2, fdr = 0.1, v = 1)
derandomKnock_regression(data_handler.grouped_X[state], data_handler.y[state], S)

full_unit_coef, full_recovered_coef = SIS_grpLasso_KKT(data_handler.X[state], data_handler.y[state], data_handler.standardized_grouped_X[state], data_handler.standardized_y[state], data_handler.grouped_X[state], data_handler.X_std[state], data_handler.y_std[state], alpha = 0.3, intercept_length = 7)
