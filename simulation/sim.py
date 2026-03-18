import os
import argparse
import numpy as np
import pandas as pd
from scipy.linalg import svd
import celer
from knockpy import KnockoffFilter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, average_precision_score, precision_recall_curve

def generate_data(n, num_group, group_size, nval, rho=0, s=10, beta_type='EvenOnes', snr=1):
    p = num_group * group_size
    x = np.random.normal(size=(n, p))
    xval = np.random.normal(size=(nval, p))

    if rho != 0:
        inds = np.arange(p)
        Sigma = rho ** np.abs(np.subtract.outer(inds, inds))
        u, d, vt = svd(Sigma)
        Sigma_half = u @ np.sqrt(np.diag(d)) @ vt
        x = x @ Sigma_half
        xval = xval @ Sigma_half
    else:
        Sigma = np.eye(p)

    beta = np.zeros((num_group, group_size))
    if beta_type == 'EvenOnes':
        beta[np.round(np.linspace(0, num_group - 1, s)).astype(int), :] = 1
    elif beta_type == 'FirstOnes':
        for c in range(group_size):
            beta[:s, c] = 1
    elif beta_type == 'LinearDec':
        for c in range(group_size):
            beta[:s, c] = np.linspace(10, 0.5, s)
    elif beta_type == 'ExpDecay':
        for c in range(group_size):
            beta[:s, c] = 1
            beta[s:, c] = 0.5 ** np.arange(1, num_group - s + 1)
    beta = beta.flatten()
    vmu = beta.T @ Sigma @ beta
    sigma = np.sqrt(vmu / snr)

    y = x @ beta + np.random.normal(scale=sigma, size=n)
    yval = xval @ beta + np.random.normal(scale=sigma, size=nval)

    return x, y, xval, yval, Sigma, beta, sigma

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
        fitted_estimator = clone(estimator).fit(X_selected, y)
    else:
        fitted_estimator = None
        if verbose:
            print("No features selected. Estimator not fitted.")

    return sorted(selected_groups), sorted(selected_features), best_score, fitted_estimator, selection_score

def run_simulation(n, num_group, group_size, nval, nrep=10, rho=0, s=10, alpha = 1.0, beta_type='EvenOnes', snr=1, seed=None, results_dir=None):
    """Run the simulation with the specified configuration."""
    if seed is not None:
        np.random.seed(seed)

    N = 3

    results = {
        "err_train": np.full((nrep, N), np.nan),
        "err_val": np.full((nrep, N), np.nan),
        "nonzero": np.full((nrep, N), np.nan),
        "avg_prec": np.full((nrep, N), np.nan)
    }
    precision_records = {
        "GroupLasso": [],
        "GroupKnock": [],
        "SequentialFeatureSelection": []
    }
    recall_records = {
        "GroupLasso": [],
        "GroupKnock": [],
        "SequentialFeatureSelection": []
    }
    estimator = LinearRegression()
    data = []
    i = 0
    while i < nrep:
        x, y, xval, yval, Sigma, beta, sigma = generate_data(
            n, num_group, group_size, nval, rho=rho, s=s, beta_type=beta_type, snr=snr
        )
        y_true_groups = (np.sum(beta.reshape(-1, group_size), axis=1) != 0).astype(int)
        
        # Model 0: Group Lasso
        j = 0
        selection_score0 = np.zeros(num_group, dtype=float)
        for alpha_ in np.geomspace(alpha * 1e-3, alpha, num=10):
            grpLasso_model = celer.GroupLasso(groups=group_size, alpha=alpha_, tol=1e-4, fit_intercept=True)
            grpLasso_model.fit(x, y)
            selection_score0[(np.sum(grpLasso_model.coef_.reshape(-1, group_size), axis=1) != 0)] = alpha_
        y_pred_train0 = grpLasso_model.predict(x)
        y_pred_val0 = grpLasso_model.predict(xval)
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
        
        # Model 1: Group Knockoffs
        j = 1
        median_correlations = []
        for group_idx in range(num_group):
            # Define the feature indices for the current group
            feature_indices = range(group_idx * group_size, (group_idx + 1) * group_size)
            # Extract the features for the current group
            x_group = x[:, feature_indices]
            # Compute correlations with y
            correlations = np.array([np.corrcoef(x_group[:, feature], y)[0, 1] for feature in range(group_size)])
            # Compute the median correlation for the group
            median_corr = np.abs(np.nanmedian(correlations))  # Using absolute value to consider magnitude
            median_correlations.append(median_corr)
        median_correlations = np.array(median_correlations)
        screened_num_group = np.min([np.max([(n - 1) // group_size, selected_k_groups]), num_group])
        indices_screened_groups = np.argsort(median_correlations)[-screened_num_group:]
        indices_screened_features = []
        for idx in indices_screened_groups:
            indices_screened_features.extend(list(range(idx * group_size, idx * group_size + group_size)))
        indices_screened_features = np.array(indices_screened_features)
        kfilter = KnockoffFilter(ksampler = 'gaussian', fstat = 'lasso')
        kfilter.forward(x[:, indices_screened_features], y, groups = np.repeat(np.arange(1, screened_num_group + 1, 1), group_size), fdr = 0.1)
        if np.sum(kfilter.W != 0) == 0:
            continue
        indices_k_groups = np.argsort(kfilter.W)[-np.min([selected_k_groups, screened_num_group]):][::-1]
        indices_features = []
        for idx in indices_k_groups:
            indices_features.extend(list(range(idx * group_size, idx * group_size + group_size)))
        GroupKnock = LinearRegression().fit(x[:, indices_screened_features[indices_features]], y)
        selection_score1 = np.zeros(num_group, dtype=float)
        selection_score1[np.isin(np.arange(num_group), indices_screened_groups, invert=True)] = np.min(kfilter.W)
        selection_score1[indices_screened_groups] = kfilter.W
        y_pred_train1 = GroupKnock.predict(x[:, indices_screened_features[indices_features]])
        y_pred_val1 = GroupKnock.predict(xval[:, indices_screened_features[indices_features]])
        results["err_train"][i, j] = mean_squared_error(y, y_pred_train1)
        results["err_val"][i, j] = mean_squared_error(yval, y_pred_val1)
        results["nonzero"][i, j] = np.sum(grpLasso_model.coef_ != 0) // group_size
        ap1 = average_precision_score(y_true_groups, selection_score1)
        results["avg_prec"][i, j] = ap1
        precision1, recall1, thresholds1 = precision_recall_curve(y_true_groups, selection_score1)
        precision1_str = ",".join(map(str, precision1))
        recall1_str = ",".join(map(str, recall1))
        precision_records["GroupKnock"].append(precision1_str)
        recall_records["GroupKnock"].append(recall1_str)
        
        # Model 2: Sequential Group Feature Selection
        j = 2
        _, forward_features, _, forward_estimator, selection_score2 = sequential_group_feature_selection(
            estimator=estimator,
            X=x,
            y=y,
            group_size=group_size,
            k_groups=selected_k_groups,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=0
        )
        y_pred_train2 = forward_estimator.predict(x[:, forward_features])
        y_pred_val2 = forward_estimator.predict(xval[:, forward_features])
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
        
        for j in range(N):
            if j == 0: 
                model_name = 'GroupLasso'
            elif j == 1: 
                model_name = 'GroupKnock'
            elif j == 2: 
                model_name = 'SequentialFeatureSelection'
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
    parser.add_argument('--n', type=int, default=100, help='Number of training samples.')
    parser.add_argument('--num_group', type=int, default=300, help='Number of groups.')
    parser.add_argument('--group_size', type=int, default=3, help='Size of each group.')
    parser.add_argument('--nval', type=int, default=100, help='Number of validation samples.')
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
                    n=args.n,
                    num_group=args.num_group,
                    group_size=args.group_size,
                    nval=args.nval,
                    nrep=args.nrep,
                    rho=rho,
                    s=args.s,
                    alpha=args.alpha, 
                    beta_type=beta_type,
                    snr=snr,
                    seed=args.seed,
                    results_dir=args.results_dir
                )
                
if __name__ == "__main__":
    main()