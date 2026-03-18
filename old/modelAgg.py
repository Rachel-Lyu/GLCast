## Aggregation of informative streams or sibling streams.

# def sort_states(arr, ref_smooth, target_state_idx, start, end):
#     Xty = []
#     for state_idx in range(len(state_list)):
#         arr_ = arr[state_idx, start:end]
#         ref_ = ref_smooth.iloc[start:end, state_idx]
#         standardized_X, standardized_y, X_mean, X_std, y_mean, y_std = standardize_data(arr_, ref_)
#         Xty.append(np.dot(standardized_X.T, standardized_y)/(end - start))
#     Xty = np.array(Xty)
#     abs_diff_Xty = np.abs(Xty - Xty[target_state_idx])
#     Rhat = []
#     for state_idx in range(len(state_list)):
#         margin_T = np.sort(abs_diff_Xty[state_idx])[::-1][:int(training_period / 3)]
#         Rhat.append(np.sum(margin_T ** 2))
#     sorted_indices = np.argsort(Rhat)[1:]
#     return sorted_indices

# # def sort_marginal_diff(primary_state, auxiliary_state_list):
# #     primary_Xty = np.dot(primary_state.standardized_X.T, primary_state.standardized_y) / len(primary_state.standardized_y)
# #     auxiliary_state_Xty = np.array([np.dot(auxiliary_state.standardized_X.T, auxiliary_state.standardized_y) / len(auxiliary_state.standardized_y) for auxiliary_state in auxiliary_state_list])
# #     abs_diff = np.abs(auxiliary_state_Xty - primary_Xty)
# #     Rhat = []
# #     for s_idx, auxiliary_state in auxiliary_state_list:
# #         screened_largest_diff = abs_diff[np.argsort(abs_diff)[-10:]]
# #         Rhat.append(np.sum(screened_largest_diff ** 2))
# #     return np.argsort(Rhat)

def get_informative_states(primary_state):
    # informative
    return auxiliary_state_list[sort_marginal_diff(primary_state, auxiliary_state_list)[:5]]
    # sibling
    for sublist in mega_list:
        if primary_state in sublist:
            return [s for s in sublist if s != primary_state]

informative_state_list = get_informative_states()

def Q_aggregation(primary_state, informative_state_list):
    states_unit_coef = np.zeros((len(informative_state_list) + 1, data_handler.nV))
    states_recovered_coef = np.zeros((len(informative_state_list) + 1, data_handler.nV))
    state = primary_state
    full_unit_coef, full_recovered_coef = single_model(data_handler.X[state], data_handler.y[state], data_handler.standardized_grouped_X[state], data_handler.standardized_y[state], data_handler.grouped_X[state], data_handler.X_std[state], data_handler.y_std[state], alpha = 0.3, intercept_length = 7)
    states_unit_coef[0] = full_unit_coef
    states_recovered_coef[0] = full_recovered_coef
    for s_idx, informative_state in enumerate(informative_state_list):
        state = informative_state
        full_unit_coef, full_recovered_coef = single_model(data_handler.X[state], data_handler.y[state], data_handler.standardized_grouped_X[state], data_handler.standardized_y[state], data_handler.grouped_X[state], data_handler.X_std[state], data_handler.y_std[state], alpha = 0.3, intercept_length = 7)
        states_unit_coef[s_idx + 1] = full_unit_coef
        states_recovered_coef[s_idx + 1] = full_recovered_coef
    B = np.copy(states_recovered_coef).T
    

# def fit_state_model(arr, ref_smooth, target_state_idx, alp, start, end):
#     SIS_v_idx = SIS(arr, ref_smooth, target_state_idx, start, end)
#     X_train_grp = get_X_grp(arr, [target_state_idx], start, end, selected_indices = SIS_v_idx)
#     y_train = np.array(ref_smooth.iloc[(start + lag_max): end, target_state_idx])
#     state_model, state_coef, state_intercept = train_grpLasso(X_train_grp, y_train, lag_set, alpha = alp, len_itc = len_itc)
#     feature_abs = np.abs(state_model.coef_.reshape((-1, group_size))).mean(axis = 1)
#     non_zero_coefs = np.where(feature_abs != 0)[0]
#     sel_predictors = SIS_v_idx[non_zero_coefs]
#     while len(sel_predictors) > 5 and alp < 1.:
#         alp = np.round(alp + 0.1, 1)
#         state_model, state_coef, state_intercept = train_grpLasso(X_train_grp, y_train, lag_set, alpha = alp, len_itc = len_itc)
#         feature_abs = np.abs(state_model.coef_.reshape((-1, group_size))).mean(axis = 1)
#         non_zero_coefs = np.where(feature_abs != 0)[0]
#         sel_predictors = SIS_v_idx[non_zero_coefs]
#     while len(sel_predictors) < 2 and alp > 0.1:
#         alp = np.round(alp - 0.1, 1)
#         if alp < 0.1:
#             alp = 0.05
#         state_model, state_coef, state_intercept = train_grpLasso(X_train_grp, y_train, lag_set, alpha = alp, len_itc = len_itc)
#         feature_abs = np.abs(state_model.coef_.reshape((-1, group_size))).mean(axis = 1)
#         non_zero_coefs = np.where(feature_abs != 0)[0]
#         sel_predictors = SIS_v_idx[non_zero_coefs]
#     coef_idx = np.array([list(range(group_size*ii, group_size*(ii+1))) for ii in non_zero_coefs]).flatten()
#     return state_model.coef_[coef_idx], state_coef[coef_idx], state_intercept, sel_predictors, alp

def TransGroupLasso(arr, ref_smooth, target_state_idx, alp, start, end, n_candidate, len_theta = 30, total_step = 10):
    sorted_states_indices = sort_states(arr, ref_smooth, target_state_idx, start, end)[:n_candidate]
    candidate_coef = np.zeros((n_candidate+1, group_size*arr.shape[2]))
    state_model_coef, _, _, sel_predictors, alp_ = fit_state_model(arr, ref_smooth, target_state_idx, alp, start, end - len_theta)
    coef_idx = np.array([list(range(group_size*ii, group_size*(ii+1))) for ii in sel_predictors]).flatten()
    candidate_coef[0, coef_idx] = state_model_coef
    
    for c_idx, candidate in enumerate(sorted_states_indices):
        state_model_coef, _, _, sel_predictors, alp_ = fit_state_model(arr, ref_smooth, candidate, alp, start, end)
        coef_idx = np.array([list(range(group_size*ii, group_size*(ii+1))) for ii in sel_predictors]).flatten()
        candidate_coef[c_idx+1, coef_idx] = state_model_coef
        
    X_train_grp = get_X_grp(arr, [target_state_idx], start, end, selected_indices = range(arr.shape[2]))
    y_train = np.array(ref_smooth.iloc[end - training_period: end, target_state_idx])
    standardized_X, standardized_y, X_mean, X_std, y_mean, y_std = standardize_data(X_train_grp, y_train)
    B = np.copy(candidate_coef).T
    #Q-aggregation
    y_test = standardized_y[-len_theta:].reshape(-1,1)
    X_test = standardized_X[-len_theta:]
    # theta.hat<- exp(-colSums((y.test-X.test%*%B)^2)/2)
    theta_hat = np.exp(-np.sum((y_test - np.dot(X_test, B))**2, axis=0) / 2)
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
    #   theta.hat<- exp(-colSums((y.test-X.test%*%B)^2)/2+colSums((as.vector(X.test%*%beta)-X.test%*%B)^2)/8)
        theta_hat = np.exp(-np.sum((y_test - np.dot(X_test, B))**2, axis=0) / 2 + 
                           np.sum((np.dot(X_test, beta).reshape(-1,1) - np.dot(X_test, B))**2, axis=0) / 8)
    #   theta.hat<-theta.hat/sum(theta.hat)
        theta_hat /= np.sum(theta_hat)
    #   beta<- as.numeric(B%*%theta.hat*1/4+3/4*beta)
        beta = np.dot(B, theta_hat) * 1/4 + 3/4 * beta
    #   if(sum(abs(theta.hat-theta.old))<10^(-3)){break}
        if np.sum(np.abs(theta_hat - theta_old)) < 1e-3:
            break
    #   theta.old=theta.hat}
        theta_old = theta_hat.copy()
    coef = calculate_coefficients(beta, X_std, y_std)
    intercept = calculate_intercept(X_train_grp, y_train, coef, len_itc)
    return beta, coef, intercept
# #Q-aggregation
# theta.hat<- exp(-colSums((y.test-X.test%*%B)^2)/2)
# theta.hat=theta.hat/sum(theta.hat)
# theta.old=theta.hat
# beta<-as.numeric(B%*%theta.hat)
# beta.ew<-beta
# # theta.old=theta.hat
# for(ss in 1:total.step){
#     theta.hat<- exp(-colSums((y.test-X.test%*%B)^2)/2+colSums((as.vector(X.test%*%beta)-X.test%*%B)^2)/8)
#     theta.hat<-theta.hat/sum(theta.hat)
#     beta<- as.numeric(B%*%theta.hat*1/4+3/4*beta)
#     if(sum(abs(theta.hat-theta.old))<10^(-3)){break}
#     theta.old=theta.hat
# }