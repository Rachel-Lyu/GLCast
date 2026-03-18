from utils import *

Strong = ['B97', 'U07']
Weak = ['Z20', 'B34', 'R50', 'R05', 'R06', 'J12', 'J18', 'J20', 'J40', 'J21', 'J96', 'J22', 'J06', 'J98', 'J80', 'R43', 'R07', 'R68']

Flu1 = ['J09', 'J10', 'J11']
Flu2 = ['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18']
Flu3 = ['J00', 'J01', 'J02', 'J03', 'J04', 'J05', 'J06', 'J20', 'J21', 'J22', 'J40', 'R05', 'H66', 'R50', 'B97']

warnings.filterwarnings("ignore")
lag_set = np.array([14, 7, 0])
group_size = len(lag_set)
report_freq_days = 1
lag_set = (lag_set / report_freq_days).astype(int)
state_list = state_abbr

train_period = 120
validation_period = 60
intercept_length = validation_period
# test_period = 300
test_period = 180
step_size = 30

# dataset_list = ['JHUcase', 'JHUdeath', 'HHShospitalized']
dataset_list = ['HHSflu']
# method_list = ['Strong', 'Weak']
method_list = ['Flu1', 'Flu2', 'Flu3']
for dataset in dataset_list:
    target_data = pd.read_csv('../data/' + dataset + '.txt', index_col = 0)
    target_data = target_data.loc[:, state_list].astype(float)
    if report_freq_days == 1:
        target_data = target_data.apply(moving_average_smoother).iloc[6:]
    target_dates = target_data.index
    state_data_dict = {}
    for state in state_list:
        if report_freq_days == 1:
            state_data = pd.read_csv(f'../data/{state}states.csv', index_col = 0, header = 0).astype(float).apply(moving_average_smoother).iloc[6:]
        elif report_freq_days == 7:
            state_data = pd.read_csv(f'../data/{state}states.csv', index_col = 0, header = 0).astype(float).rolling(window=7, min_periods=1).sum().iloc[6:]
        state_data_dict[state] = state_data[state_data.index.isin(target_dates)]
    all_codes = state_data_dict[state_list[0]].columns.to_numpy()

    train_start_date_list = []
    validation_start_date_list = []
    test_start_date_list = []
    # train_start_date = date_after(max(min(target_data.index), min(state_data_dict[state].index)), max(lag_set) * report_freq_days)
    train_start_date = 20211101
    validation_start_date = date_after(train_start_date, train_period)
    test_start_date = date_after(validation_start_date, validation_period)
    # while test_start_date < 20220901:
    while test_start_date < 20230401:
        train_start_date_list.append(train_start_date)
        validation_start_date_list.append(validation_start_date)
        test_start_date_list.append(test_start_date)
        train_start_date = date_after(train_start_date, step_size)
        validation_start_date = date_after(train_start_date, train_period)
        test_start_date = date_after(validation_start_date, validation_period)

    data_handler = HealthDataHandler(state_list, lag_set, report_freq_days, target_data, state_data_dict)
    # for m_idx, DiagnosisCodeSet in enumerate([Strong, Strong + Weak]):
    for m_idx, DiagnosisCodeSet in enumerate([Flu1, Flu1 + Flu2, Flu1 + Flu2 + Flu3]):
        active_g_indices = np.where(np.isin(all_codes, DiagnosisCodeSet))[0]
        active_v_indices = extend_indices(active_g_indices, group_size)
        y_dict = {}
        pred_dict = {}
        pred_agg_dict = {}
        date_plot_list = []
        for state in state_list:
            y_dict[state] = []
            pred_dict[state] = []
            pred_agg_dict[state] = []
        
        selection = False
        for d_idx in range(len(train_start_date_list)):
            data_handler.get_X_y(start_date = train_start_date_list[d_idx], end_date = validation_start_date_list[d_idx], codes = all_codes)
            data_handler.get_test(start_test = validation_start_date_list[d_idx], codes = all_codes, period_length = validation_period)
            print(train_start_date_list[d_idx], '-', test_start_date_list[d_idx])
            
            # data_handler.run_all_single(intercept_length, method, alpha = 0.2, states_prev_g = states_prev_g, maxsteps = 4, states_prev_W = states_prev_W, tau = 0.5, M = 10, M_max = 30, fdr = 0.1)
            data_handler.recovered_coef = {}
            data_handler.recovered_intercept = {}
            for state in state_list:
                grouped_X = data_handler.grouped_X[state]
                y = data_handler.y[state]
                X_active = grouped_X[:, active_v_indices]
                extended_X_active = np.hstack([np.ones((grouped_X.shape[0], 1)), X_active])
                beta, _, _, _ = np.linalg.lstsq(extended_X_active, y, rcond=None)
                recovered_coef = np.zeros(grouped_X.shape[1])
                recovered_coef[active_v_indices] = beta[1:]
                recovered_intercept = recover_intercept(grouped_X, y, recovered_coef, intercept_length)
                data_handler.recovered_coef[state] = recovered_coef
                data_handler.recovered_intercept[state] = recovered_intercept
            
            data_handler.run_all_agg(intercept_length, state_method = 'marginal_diff', auxiliary_state_list = None, nStates = 5, total_step = 10, selection = selection)
            
            data_handler.get_test(start_test = test_start_date_list[d_idx], codes = all_codes, period_length = test_period)
            date_plot_list.append([datetime.strptime(str(d), '%Y%m%d') for d in data_handler.test_dates])
            for state in state_list:
                y_dict[state].append(data_handler.y_test[state])
                pred_dict[state].append(np.dot(data_handler.grouped_X_test[state], data_handler.recovered_coef[state]) + data_handler.recovered_intercept[state])
                pred_agg_dict[state].append(np.dot(data_handler.grouped_X_test[state], data_handler.recovered_coef_agg[state]) + data_handler.recovered_intercept_agg[state])

            dfout = {
                'date': date_plot_list[d_idx],
                **{f'y_{state}': y_dict[state][d_idx] for state in state_list},
                **{f'pred_{state}': pred_dict[state][d_idx] for state in state_list},
                **{f'pred_agg_{state}': pred_agg_dict[state][d_idx] for state in state_list}
            }
            dfout = pd.DataFrame(dfout)
            dfout.to_csv(f'test0206/{dataset}_{method_list[m_idx]}_{d_idx}.csv', index=False)
        # for state in state_list:
        #     plt.figure(figsize=(10, 6))
        #     for d_idx in range(len(train_start_date_list)):
        #         plt.plot(date_plot_list[d_idx], y_dict[state][d_idx], c = 'k', lw = 2)
        #         plt.plot(date_plot_list[d_idx], pred_dict[state][d_idx], alpha = 0.8, lw = 2)
        #     plt.title(state + ', single')
        #     plt.xticks(rotation = 30)
        #     plt.figure(figsize=(10, 6))
        #     for d_idx in range(len(train_start_date_list)):
        #         plt.plot(date_plot_list[d_idx], y_dict[state][d_idx], c = 'k', lw = 2)
        #         plt.plot(date_plot_list[d_idx], pred_agg_dict[state][d_idx], alpha = 0.8, lw = 2)
        #     plt.title(state + ', aggregated')
        #     plt.xticks(rotation = 30)
        
