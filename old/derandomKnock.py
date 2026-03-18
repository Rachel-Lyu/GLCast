import numpy as np
from knockpy import KnockoffFilter
from sklearn.linear_model import LinearRegression
#' @param v a positive number indicating the parameter of the base procedure.
#' @param M an integer specifying the number of knockoff copies computed (default: 30).
#' @param tau a number betweem 0 and 1 indicating the selection frequency (default: 0.5).

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
    full_curr_W = [np.zeros(X.shape[1]) for w_idx in range(len(curr_W))]
    for w_idx, W in enumerate(curr_W):
        full_curr_W[w_idx][SIS_g_indices] = W
    recovered_coef, _ = derandomKnock_regression(grouped_X, y, S_g_indices)
    recovered_intercept = recover_intercept(grouped_X, y, recovered_coef, intercept_length)
    return S_g_indices, recovered_coef, recovered_intercept, full_curr_W