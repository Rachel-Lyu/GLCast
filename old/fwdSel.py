from sklearn.linear_model import LinearRegression
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
    return active_g_indices, forward_coef, forward_intercept, recovered_intercept

# def forward_regression(X, y, k, grp, fold = 3):
#     n_features = int(X.shape[1]/grp)  # Number of predictor variables
#     selected_features = []   # List to store selected features
#     selected_features_index = []
#     for _ in range(k):
#         best_r2 = 0  # Initialize the best R^2 value
#         best_feature = None  # Initialize the best feature
#          # Iterate through all features not already selected
#         for feature in range(n_features):
#             if feature not in selected_features:
#                 features_subset = selected_features_index + [feature * grp + g for g in range(grp)]
#                 X_subset = X[:, features_subset]
#                 train_indices, test_indices = split_data_indices(X_subset.shape[0], test_ratio = 1/fold)
#                 # Fit a linear regression model with the selected features
#                 model = LinearRegression()
#                 model.fit(X_subset[train_indices], y[train_indices])
#                 # Calculate the R^2 score
#                 r2 = model.score(X_subset[test_indices], y[test_indices])
#                 # Update the best feature if necessary

#                 if r2 > best_r2:
#                     best_r2 = r2
#                     best_feature = feature
        
#         # Add the best feature to the selected list
#         selected_features.append(best_feature)
#         selected_features_index = selected_features_index + [best_feature + g for g in range(grp)]
#     return selected_features