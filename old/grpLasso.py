# SIS - grpLasso - KKT
import numpy as np
import celer
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
    return active_g_indices, full_unit_coef, full_recovered_coef, recovered_intercept
