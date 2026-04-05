import os
import argparse
import numpy as np
import pandas as pd
from scipy.linalg import svd
import celer
# from knockpy import KnockoffFilter
from sklearn.utils import check_random_state
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, average_precision_score, precision_recall_curve


def generate_data_sites(n_aux_sites, informative_sites, num_group, group_size, n_list, nval_list, rho=0, s=10, beta_type='EvenOnes', snr=1, h=6, q=20, random_state=42):
    rng = check_random_state(random_state)
    beta_list = []
    X_list = []
    y_list = []
    Xval_list = []
    yval_list = []
    
    p = num_group * group_size
    if rho != 0:
        inds = np.arange(p)
        Sigma = rho ** np.abs(np.subtract.outer(inds, inds))
        u, d, vt = svd(Sigma)
        Sigma_half = u @ np.sqrt(np.diag(d)) @ vt
    else:
        Sigma = np.eye(p)
        
    beta2D = np.zeros((num_group, group_size))
    if beta_type == 'EvenOnes':
        beta2D[np.round(np.linspace(0, num_group - 1, s)).astype(int), :] = 0.3
    elif beta_type == 'FirstOnes':
        for c in range(group_size):
            beta2D[:s, c] = 0.3
    elif beta_type == 'LinearDec':
        for c in range(group_size):
            beta2D[:s, c] = 0.3 * np.linspace(10, 0.5, s)
    elif beta_type == 'ExpDecay':
        for c in range(group_size):
            beta2D[:s, c] = 0.3
            beta2D[s:, c] = 0.3 * 0.5 ** np.arange(1, num_group - s + 1)
    beta = beta2D.flatten()
    vmu = beta.T @ Sigma @ beta
    sigma = np.sqrt(vmu / snr)
    
    beta_list.append(beta)
    
    for idx_site in range(n_aux_sites + 1):
        n = n_list[idx_site]
        nval = nval_list[idx_site]
        X = np.random.normal(size=(n, p))
        Xval = np.random.normal(size=(nval, p))
        if rho != 0:
            X = X @ Sigma_half
            Xval = Xval @ Sigma_half
        X_list.append(X)
        Xval_list.append(Xval)
        
        if idx_site == 0:
            y = X @ beta + np.random.normal(scale=sigma, size=n)
            yval = Xval @ beta + np.random.normal(scale=sigma, size=nval)
            y_list.append(y)
            yval_list.append(yval)
            
        elif idx_site in informative_sites:
            # samp0<- sample(1:p, h, replace=F)
            # W[samp0,k] <-W[samp0,k] + rep(-sig.delta1, h)
            # W[1:100,k] <-W[1:100,k] + rnorm(100, 0, h/100)
            beta_info = beta2D.copy()
            samp0 = rng.choice(num_group, size=h, replace=False)
            beta_info[samp0, :] = beta_info[samp0, :] + rng.normal(loc=0, scale=0.3, size=(h, 1))
            beta_info = beta_info.flatten()
            beta_list.append(beta_info)
            
            y = X @ beta_info + np.random.normal(scale=sigma, size=n)
            yval = Xval @ beta_info + np.random.normal(scale=sigma, size=nval)
            y_list.append(y)
            yval_list.append(yval)
            
        else:
            # samp1 <- sample(1:p, q, replace = F)
            # W[samp1,k] <- W[samp1,k] + rep(-sig.delta2,q)
            # W[1:100,k] <-W[1:100,k] + rnorm(100, 0, q/100)
            beta_uninfo = beta2D.copy()
            samp1 = rng.choice(num_group, size=q, replace=False)
            beta_uninfo[samp1, :] = beta_uninfo[samp1, :] + rng.normal(loc=0, scale=1.0, size=(q, 1))
            beta_uninfo = beta_uninfo.flatten()
            beta_list.append(beta_uninfo)
            
            y = X @ beta_uninfo + np.random.normal(scale=sigma, size=n)
            yval = Xval @ beta_uninfo + np.random.normal(scale=sigma, size=nval)
            y_list.append(y)
            yval_list.append(yval)
    
    return X_list, Xval_list, y_list, yval_list, Sigma, beta_list, sigma

def sequential_group_feature_selection(
    estimator,
    X,
    y,
    group_size,
    k_groups,
    scoring=None,
    cv=5,
    verbose=0
):
    """
    Perform Sequential Group Feature Selection.

    Parameters
    ----------
    estimator : scikit-learn estimator
        The machine learning model to evaluate feature subsets.

    X : array-like, shape (n_samples, n_features)
        The input data.

    y : array-like, shape (n_samples,)
        The target variable.

    group_size : int
        Number of elements each group.

    k_groups : int
        The number of groups to select.

    scoring : str or callable, default=None
        Scoring metric to use. If None, uses the estimator's default scorer.

    cv : int or cross-validation generator, default=5
        Number of cross-validation folds.

    verbose : int, default=0
        Verbosity level. 0 means silent, higher values increase output detail.

    Returns
    -------
    selected_groups : list
        List of selected group indices.

    selected_features : list
        List of selected feature indices.

    best_score : float
        Best cross-validation score achieved.

    fitted_estimator : estimator
        The estimator fitted on the selected feature groups.
    """
    
    num_group = X.shape[1] // group_size
    groups = np.arange(X.shape[1]).reshape(-1, group_size)
    all_group_indices = set(range(num_group))
    selected_groups = set()
    remaining_groups = all_group_indices.copy()
    best_score = -np.inf if scoring is not None else None
    
    selection_score = np.zeros(num_group, dtype=int)

    for i in range(k_groups):
        candidates = remaining_groups

        scores = {}
        for group in candidates:
            subset_groups = selected_groups.union({group})

            # Aggregate feature indices from the selected groups
            subset_features = []
            for grp in subset_groups:
                subset_features.extend(groups[grp])

            if not subset_features:
                continue

            X_subset = X[:, subset_features]
            estimator_clone = clone(estimator)
            cv_scores = cross_val_score(estimator_clone, X_subset, y, cv=cv, scoring=scoring)
            avg_score = np.mean(cv_scores)
            scores[group] = avg_score

            if verbose > 1:
                print(f"Evaluated groups {sorted(subset_groups)}: Score = {avg_score:.4f}")

        if not scores:
            break

        # Select the group with the best score
        if scoring is not None:
            best_group = max(scores, key=scores.get)
            best_group_score = scores[best_group]
        else:
            # If no scoring, just pick the first group
            best_group = next(iter(scores))
            best_group_score = None

        selected_groups.add(best_group)
        remaining_groups.remove(best_group)
        selection_score[best_group] = k_groups - i
        if verbose:
            print(f"Step {i+1}: Added group {best_group} with score {best_group_score:.4f}")

        if scoring is not None and best_group_score > best_score:
            best_score = best_group_score

    # Aggregate selected feature indices
    selected_features = []
    for grp in selected_groups:
        selected_features.extend(groups[grp])

    # Fit the estimator on the selected features
    if selected_features:
        X_selected = X[:, selected_features]
    else:
        X_selected = X
        selected_features = np.arange(X.shape[1])
        selected_groups = np.arange(num_group)
    fitted_estimator = clone(estimator).fit(X_selected, y)

    return sorted(selected_groups), sorted(selected_features), best_score, fitted_estimator, selection_score

def sort_marginal_diff(X_list, y_list):
    Xty_list = [np.dot(X_list[i].T, y_list[i]) / len(y_list[i]) for i in range(len(y_list))]
    abs_diff = np.abs(Xty_list[1:] - Xty_list[0])
    Rhat = []
    for s_idx in range(1, len(y_list)):
        abs_diff_s = abs_diff[s_idx - 1]
        screened_largest_diff = abs_diff_s[np.argsort(abs_diff_s)[-int(len(y_list[s_idx])/3):]]
        Rhat.append(np.sum(screened_largest_diff ** 2))
    return np.argsort(Rhat) + 1

def Q_aggregation(B, X_test, y_test, total_step = 10, selection = False):
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

def Q_aggregation_cv(B, X, y, total_step = 10, selection = False):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    theta_hats = []
    betas = []
    cv_errors = []
    
    fold = 1
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        theta_hat, beta = Q_aggregation(
            B,
            X_val,
            y_val,
            total_step=total_step,
            selection=selection
        )
        theta_hats.append(theta_hat)
        betas.append(beta)
        y_pred = X_val @ beta
        mse = mean_squared_error(y_val, y_pred)
        cv_errors.append(mse)
    theta_hats = np.array(theta_hats)
    betas = np.array(betas) 
    inverse_mse = 1 / np.array(cv_errors)
    weights = inverse_mse / inverse_mse.sum()
    weighted_avg_theta_hat = np.average(theta_hats, axis=0, weights=weights)
    weighted_avg_beta = np.average(betas, axis=0, weights=weights)
    return weighted_avg_theta_hat, weighted_avg_beta

def forward_all(X_list, y_list, group_size, k_groups):
    estimator = LinearRegression()
    B = np.zeros((len(y_list), X_list[0].shape[1] + 1))
    for i in range(len(y_list)):
        X = X_list[i]
        y = y_list[i]
        _, forward_features, _, forward_estimator, _ = sequential_group_feature_selection(estimator=estimator, X=X, y=y, group_size=group_size, k_groups=k_groups, scoring='neg_mean_squared_error', cv=5, verbose=0)
        B[i, forward_features] = forward_estimator.coef_
        B[i, -1] = forward_estimator.intercept_
    return B

def grpLasso_all(X_list, y_list, fitted_model = None, group_size = None, alpha = 1.0):
    B = np.zeros((len(y_list), X_list[0].shape[1] + 1))
    if fitted_model is None:
        grpLasso_model = celer.GroupLasso(groups=group_size, alpha=alpha, tol=1e-4, fit_intercept=True)
    else: 
        grpLasso_model = clone(fitted_model)
    for i in range(len(y_list)):
        X = X_list[i]
        y = y_list[i]
        grpLasso_model.fit(X, y)
        B[i, :-1] = grpLasso_model.coef_
        B[i, -1] = grpLasso_model.intercept_
    return B

def run_simulation(n_aux_sites, informative_sites, num_group, group_size, n_list, nval_list, nrep=10, rho=0, s=10, alpha = 1.0, beta_type='EvenOnes', snr=1, h=6, q=20, n_agg = 5, seed=None, results_dir=None):
    """Run the simulation with the specified configuration."""
    if seed is not None:
        np.random.seed(seed)

    N = 4

    results = {
        "err_train": np.full((nrep, N), np.nan),
        "err_val": np.full((nrep, N), np.nan),
        "nonzero": np.full((nrep, N), np.nan),
        "avg_prec": np.full((nrep, N), np.nan)
    }
    precision_records = {
        "GroupLasso": [],
        # "GroupKnock": [],
        "SequentialFeatureSelection": [],
        "Agg_GroupLasso": [],
        # "Agg_GroupKnock": [],
        "Agg_SequentialFeatureSelection": []
    }
    recall_records = {
        "GroupLasso": [],
        # "GroupKnock": [],
        "SequentialFeatureSelection": [],
        "Agg_GroupLasso": [],
        # "Agg_GroupKnock": [],
        "Agg_SequentialFeatureSelection": []
    }
    estimator = LinearRegression()
    data = []
    i = 0
    while i < nrep:
        X_list, Xval_list, y_list, yval_list, Sigma, beta_list, sigma = generate_data_sites(
            n_aux_sites, informative_sites, num_group, group_size, n_list, nval_list, 
            rho, s, beta_type, snr, h, q, random_state=seed)
        y_true_groups = (np.sum(beta_list[0].reshape(-1, group_size), axis=1) != 0).astype(int)
        
        # non_agg
        X = X_list[0]
        Xval = Xval_list[0]
        y = y_list[0]
        yval = yval_list[0]
        n = n_list[0]
        # Model 0: Group Lasso
        j = 0
        selection_score0 = np.zeros(num_group, dtype=float)
        for alpha_ in np.geomspace(alpha * 1e-3, alpha, num=10):
            grpLasso_model = celer.GroupLasso(groups=group_size, alpha=alpha_, tol=1e-4, fit_intercept=True)
            grpLasso_model.fit(X, y)
            selection_score0[(np.sum(grpLasso_model.coef_.reshape(-1, group_size), axis=1) != 0)] = alpha_
        y_pred_train0 = grpLasso_model.predict(X)
        y_pred_val0 = grpLasso_model.predict(Xval)
        results["err_train"][i, j] = mean_squared_error(y, y_pred_train0)
        results["err_val"][i, j] = mean_squared_error(yval, y_pred_val0)
        results["nonzero"][i, j] = np.sum(grpLasso_model.coef_ != 0) // group_size
        ap0 = average_precision_score(y_true_groups, selection_score0)
        results["avg_prec"][i, j] = ap0
        precision0, recall0, thresholds0 = precision_recall_curve(y_true_groups, selection_score0)
        precision0_str = ",".join(map(str, precision0))
        recall0_str = ",".join(map(str, recall0))
        precision_records["GroupLasso"].append(precision0_str)
        recall_records["GroupLasso"].append(recall0_str)
        selected_k_groups = np.sum(grpLasso_model.coef_ != 0) // group_size
        
        # # Model 1: Group Knockoffs
        # j = 1
        # median_correlations = []
        # for group_idx in range(num_group):
        #     # Define the feature indices for the current group
        #     feature_indices = range(group_idx * group_size, (group_idx + 1) * group_size)
        #     # Extract the features for the current group
        #     X_group = X[:, feature_indices]
        #     # Compute correlations with y
        #     correlations = np.array([np.corrcoef(X_group[:, feature], y)[0, 1] for feature in range(group_size)])
        #     # Compute the median correlation for the group
        #     median_corr = np.abs(np.nanmedian(correlations))  # Using absolute value to consider magnitude
        #     median_correlations.append(median_corr)
        # median_correlations = np.array(median_correlations)
        # screened_num_group = np.min([np.max([(n - 1) // group_size, selected_k_groups]), num_group])
        # indices_screened_groups = np.argsort(median_correlations)[-screened_num_group:]
        # indices_screened_features = []
        # for idx in indices_screened_groups:
        #     indices_screened_features.extend(list(range(idx * group_size, idx * group_size + group_size)))
        # indices_screened_features = np.array(indices_screened_features)
        # kfilter = KnockoffFilter(ksampler = 'gaussian', fstat = 'lasso')
        # kfilter.forward(X[:, indices_screened_features], y, groups = np.repeat(np.arange(1, screened_num_group + 1, 1), group_size), fdr = 0.1)
        # if np.sum(kfilter.W != 0) == 0:
        #     continue
        # indices_k_groups = np.argsort(kfilter.W)[-np.min([selected_k_groups, screened_num_group]):][::-1]
        # indices_features = []
        # for idx in indices_k_groups:
        #     indices_features.extend(list(range(idx * group_size, idx * group_size + group_size)))
        # GroupKnock = LinearRegression().fit(X[:, indices_screened_features[indices_features]], y)
        # selection_score1 = np.zeros(num_group, dtype=float)
        # selection_score1[np.isin(np.arange(num_group), indices_screened_groups, invert=True)] = np.min(kfilter.W)
        # selection_score1[indices_screened_groups] = kfilter.W
        # y_pred_train1 = GroupKnock.predict(X[:, indices_screened_features[indices_features]])
        # y_pred_val1 = GroupKnock.predict(Xval[:, indices_screened_features[indices_features]])
        # results["err_train"][i, j] = mean_squared_error(y, y_pred_train1)
        # results["err_val"][i, j] = mean_squared_error(yval, y_pred_val1)
        # results["nonzero"][i, j] = np.sum(GroupKnock.coef_ != 0) // group_size
        # ap1 = average_precision_score(y_true_groups, selection_score1)
        # results["avg_prec"][i, j] = ap1
        # precision1, recall1, thresholds1 = precision_recall_curve(y_true_groups, selection_score1)
        # precision1_str = ",".join(map(str, precision1))
        # recall1_str = ",".join(map(str, recall1))
        # precision_records["GroupKnock"].append(precision1_str)
        # recall_records["GroupKnock"].append(recall1_str)
        
        # Model 1: Sequential Group Feature Selection
        j = 1
        _, forward_features, _, forward_estimator, selection_score2 = sequential_group_feature_selection(
            estimator=estimator,
            X=X,
            y=y,
            group_size=group_size,
            k_groups=selected_k_groups,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=0
        )
        y_pred_train2 = forward_estimator.predict(X[:, forward_features])
        y_pred_val2 = forward_estimator.predict(Xval[:, forward_features])
        results["err_train"][i, j] = mean_squared_error(y, y_pred_train2)
        results["err_val"][i, j] = mean_squared_error(yval, y_pred_val2)
        results["nonzero"][i, j] = np.sum(forward_estimator.coef_ != 0) // group_size
        ap2 = average_precision_score(y_true_groups, selection_score2)
        results["avg_prec"][i, j] = ap2
        precision2, recall2, thresholds2 = precision_recall_curve(y_true_groups, selection_score2)
        precision2_str = ",".join(map(str, precision2))
        recall2_str = ",".join(map(str, recall2))
        precision_records["SequentialFeatureSelection"].append(precision2_str)
        recall_records["SequentialFeatureSelection"].append(recall2_str)
        
        # agg
        informative_sites_inferred = sort_marginal_diff(X_list, y_list)[:n_agg]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        # Model 2: Trans Group Lasso
        j = 2
        transB = grpLasso_all([X_train] + [X_list[idx] for idx in informative_sites_inferred],
                            [y_train] + [y_list[idx] for idx in informative_sites_inferred],
                            fitted_model = grpLasso_model)
        theta_hat, beta_transGrpLasso = Q_aggregation(np.transpose(transB), np.c_[X_test, np.ones(X_test.shape[0])], y_test)
        print(theta_hat)
        y_pred_train3 = np.dot(X, beta_transGrpLasso[:-1]) + beta_transGrpLasso[-1]
        y_pred_val3 = np.dot(Xval, beta_transGrpLasso[:-1]) + beta_transGrpLasso[-1]
        results["err_train"][i, j] = mean_squared_error(y, y_pred_train3)
        results["err_val"][i, j] = mean_squared_error(yval, y_pred_val3)
        results["nonzero"][i, j] = np.sum((beta_transGrpLasso[:-1]) != 0) // group_size
        
        # Model 3: Trans Forward
        j = 3
        transB = forward_all([X_train] + [X_list[idx] for idx in informative_sites_inferred],
                            [y_train] + [y_list[idx] for idx in informative_sites_inferred],
                            group_size, k_groups = selected_k_groups)
        theta_hat, beta_transForward = Q_aggregation(np.transpose(transB), np.c_[X_test, np.ones(X_test.shape[0])], y_test)
        y_pred_train4 = np.dot(X, beta_transForward[:-1]) + beta_transForward[-1]
        y_pred_val4 = np.dot(Xval, beta_transForward[:-1]) + beta_transForward[-1]
        results["err_train"][i, j] = mean_squared_error(y, y_pred_train4)
        results["err_val"][i, j] = mean_squared_error(yval, y_pred_val4)
        results["nonzero"][i, j] = np.sum((beta_transForward[:-1]) != 0) // group_size
        
        for j in range(N):
            if j == 0: 
                model_name = 'GroupLasso'
            # elif j == 1: 
            #     model_name = 'GroupKnock'
            elif j == 1: 
                model_name = 'SequentialFeatureSelection'
            elif j == 2: 
                model_name = 'TransGroupLasso'
            elif j == 3: 
                model_name = 'TransForwardSelection'
            data.append({
                'model': model_name,
                'err_train': results["err_train"][i, j],
                'err_val': results["err_val"][i, j],
                'nonzero': results["nonzero"][i, j],
                'avg_prec': results["avg_prec"][i, j],
                'rho': rho,
                'snr': snr,
                'beta_type': beta_type
            })
        if results_dir is not None:
            df = pd.DataFrame(data)
            df_file = f"beta_{beta_type}_rho_{rho}_snr_{snr}.csv"
            os.makedirs(results_dir, exist_ok=True)
            df.to_csv(os.path.join(results_dir, df_file), index=False)
        for model_name in precision_records.keys():
            precision_file = f"beta_{beta_type}_rho_{rho}_snr_{snr}_{model_name}_precision.txt"
            recall_file    = f"beta_{beta_type}_rho_{rho}_snr_{snr}_{model_name}_recall.txt"
            precision_path = os.path.join(results_dir, precision_file)
            recall_path    = os.path.join(results_dir, recall_file)
            with open(precision_path, 'w') as pf:
                for precision_line in precision_records[model_name]:
                    pf.write(precision_line + "\n")
            with open(recall_path, 'w') as rf:
                for recall_line in recall_records[model_name]:
                    rf.write(recall_line + "\n")
        i += 1
    return results

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Group Feature Selection Simulations.")
    parser.add_argument('--n_aux_sites', type=int, default=20, help='Number of auxiliary sites.')
    parser.add_argument('--informative_sites', type=int, nargs='+', default=np.arange(1, 8), help='Indices (>0) of informative sites.')
    parser.add_argument('--num_group', type=int, default=200, help='Number of groups.')
    parser.add_argument('--group_size', type=int, default=3, help='Size of each group.')
    parser.add_argument('--n_list', type=int, nargs='+', default=np.repeat(100, 21), help='List of numbers of samples.')
    parser.add_argument('--nval_list', type=int, nargs='+', default=np.repeat(100, 21), help='List of numbers of validation samples.')
    parser.add_argument('--nrep', type=int, default=50, help='Number of repetitions.')
    parser.add_argument('--rho', type=float, nargs='+', default=[0.7], help='List of correlation coefficients.')
    parser.add_argument('--snr', type=float, nargs='+', default=[1], help='List of signal-to-noise ratios.')
    parser.add_argument('--beta_type', type=str, nargs='+', default=['EvenOnes'], help='List of beta types.')
    parser.add_argument('--s', type=int, default=10, help='Number of non-zero groups.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Constant that multiplies the penalty term in celer.GroupLasso.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--results_dir', type=str, default='simulation_results', help='Directory to save results.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    os.makedirs(args.results_dir, exist_ok=True)
    for beta_type in args.beta_type:
        for rho in args.rho:
            for snr in args.snr:
                run_simulation(
                    n_aux_sites=args.n_aux_sites, 
                    informative_sites=args.informative_sites, 
                    num_group=args.num_group,
                    group_size=args.group_size,
                    n_list=args.n_list, 
                    nval_list=args.n_list,
                    nrep=args.nrep,
                    rho=rho,
                    s=args.s,
                    alpha=args.alpha, 
                    beta_type=beta_type,
                    h=6,
                    q=2*args.s,
                    n_agg=5,
                    snr=snr,
                    seed=args.seed,
                    results_dir=args.results_dir
                )
                
if __name__ == "__main__":
    main()