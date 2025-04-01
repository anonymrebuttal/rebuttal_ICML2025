import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from scipy.stats import multivariate_normal
import itertools
from collections import defaultdict
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
import quantile_forest as qf
from sklearn.linear_model import LinearRegression
from scipy import stats
import random
from scipy.stats import mstats
import matplotlib.pyplot as plt


def plot_multiple_weighted_qq(all_scores1, all_weights1, all_scores2, all_weights2):
    def weighted_quantiles(values, weights, quantiles):
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]
        cumulative_weights = np.cumsum(weights)
        cumulative_weights /= cumulative_weights[-1]
        return np.interp(quantiles, cumulative_weights, values)

    num_pairs = all_scores1.shape[0]
    num_points = all_scores1.shape[1]

    plt.figure(figsize=(4, 4))

    for i in range(num_pairs):
        scores1 = all_scores1[i]
        scores2 = all_scores2[i]
        weights1 = all_weights1[i][:-1]  # remove ∞
        weights2 = all_weights2[i][:-1]  # remove ∞

        # Normalize weights
        weights1 = weights1 / np.sum(weights1)
        weights2 = weights2 / np.sum(weights2)

        # Compute quantiles (avoid 0 and 1)
        quantile_levels = np.linspace(0, 1, num_points, endpoint=False)[1:]

        q1 = weighted_quantiles(scores1, weights1, quantile_levels)
        q2 = weighted_quantiles(scores2, weights2, quantile_levels)

        plt.plot(q1, q2, alpha=0.5)

    max_val = max(np.max(all_scores1), np.max(all_scores2))
    min_val = min(np.min(all_scores1), np.min(all_scores2))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    plt.xlabel('Estimated Nonconformity Score Distribution')
    plt.ylabel('Real Nonconformity Score Distribution')
    plt.title('{} Q-Q Plot for Different Experiments'.format(num_pairs))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_exp(seed, mu, Sigma, beta, cal_num, M_index, obs_index, alpha,
            test_num, M_cond, reweight=False, estimation_reweight=False, estimation_random_noise=False, random_noise=0.5):

    def score_fn(reg, x, y):
        predictions = predict_basemodel(reg, x, 'Quantiles', 'Linear', alpha)
        return np.maximum(predictions['y_inf'] - y, y - predictions['y_sup'])

    def score_inv_fn(reg, s, x):
        predictions = predict_basemodel(reg, x, 'Quantiles', 'Linear', alpha)
        return [predictions['y_inf'] - s, predictions['y_sup'] + s]

    master_rng = random.Random(seed)
    seeds = [master_rng.randint(1, 10**5) for _ in range(10)]

    dataloader_train = gaussian_linear(mu, Sigma, beta, seed=seeds[0], n=500)
    dataloader_train.generate_mcar(missing_prob=0.5)
    Y_train = dataloader_train.Y.copy()

    dataloader_calib = gaussian_linear(mu, Sigma, beta, seed=seeds[1], n=cal_num)
    dataloader_calib.generate_mcar(missing_prob=0.5)
    Y_calib = dataloader_calib.Y.copy()

    dataloader_train.all_imputation(mean_shift=0.5, mean_scale=1.5, var_perturb=0.2)
    dataloader_calib.all_imputation(mean_shift=0.5, mean_scale=1.5, var_perturb=0.2)

    dataloader_train.impute_with_perturbation()  # the train dataset is imputed by a shifted imputation function
    dataloader_calib.impute_with_perturbation()  # the Cal dataset is imputed by a shifted imputation function

    mask_class_data = Data_for_mask(
        np.column_stack([dataloader_train.X_imputed, dataloader_train.Y]),
        dataloader_train.M,
        batch_size=256,
        contain_y=True,
        dataset='gaussian_linear')

    pre_imputed_calib_X = dataloader_calib.X_imputed.copy()
    pre_imputed_calib_X[:, M_index] = np.nan
    M_calib = dataloader_calib.M
    reg = fit_basemodel(dataloader_train.X_imputed[:,obs_index], Y_train, target='Quantiles',
                                          basemodel='Linear', alpha=alpha,
                                          params_basemodel={'cores':-1})

    score_calib = score_fn(reg, pre_imputed_calib_X[:, obs_index], Y_calib)
    dataloader_test = gaussian_linear(mu, Sigma, beta, seed=seeds[2], n=test_num)
    Y_test = dataloader_test.Y.copy()

    if reweight:
        if estimation_reweight:
            lr_esti = Miss_proba_simple_model(mask_class_data, model='histgbdt', epoch=500)
            lr_esti.train()

            calib_pre_imputed_X = dataloader_calib.X_imputed.copy()
            calib_pre_imputed_X[:,M_index] = np.nan
            calib_likelihood_ratio = lr_esti.eval(np.hstack((pre_imputed_calib_X, Y_calib.reshape(-1, 1))))
            errors = lr_esti.error(np.hstack((pre_imputed_calib_X, Y_calib.reshape(-1, 1))),
                                   M_calib,
                                   M_cond
                                   )

            if estimation_random_noise:
                rng = np.random.default_rng(seeds[4])
                log_lr = np.log(np.maximum(calib_likelihood_ratio, 1e-12))
                noisy_log_lr = log_lr + rng.normal(0, random_noise, size=log_lr.shape)
                calib_likelihood_ratio =  np.exp(noisy_log_lr)

            true_calib_likelihood_ratio = []
            for i, cal_point in enumerate(dataloader_calib.X_imputed):
                true_calib_likelihood_ratio.append(dataloader_calib.likelihood_ratio_estimation(cal_point[obs_index],Y_calib[i],obs_index))
            true_calib_likelihood_ratio = np.array(true_calib_likelihood_ratio)

            test_missing = dataloader_test.X.copy()
            test_missing[:, M_index] = np.nan
            test_likelihood_ratio = lr_esti.eval(np.hstack((test_missing, Y_test.reshape(-1, 1))))

        else:
            calib_likelihood_ratio = []
            for i, calib_point in enumerate(dataloader_calib.X_imputed):
                calib_likelihood_ratio.append(dataloader_calib.likelihood_ratio_estimation(calib_point[obs_index],Y_calib[i],obs_index))
            calib_likelihood_ratio = np.array(calib_likelihood_ratio)
            test_likelihood_ratio = []
            for i, test_point in enumerate(dataloader_test.X):
                test_likelihood_ratio.append(dataloader_calib.likelihood_ratio_estimation(test_point[obs_index],Y_test[i],obs_index))

        test_likelihood_ratio = np.array(test_likelihood_ratio)
        test_len = test_likelihood_ratio.shape[0]
        calib_likelihood_ratio = np.asarray(calib_likelihood_ratio).reshape(1, -1)
        test_likelihood_ratio = np.asarray(test_likelihood_ratio).reshape(-1, 1)

        combined_y = np.hstack([calib_likelihood_ratio.repeat(len(test_likelihood_ratio), axis=0), test_likelihood_ratio])
        weights_y = combined_y / combined_y.sum(axis=1, keepdims=True)

        expanded_array_y = np.tile(score_calib, (test_len, 1))
        expanded_array_y = np.hstack(
            (expanded_array_y, np.full((test_len, 1), np.nan)))

        quantile_score_calib = weighted_quantile_matrix(expanded_array_y, weights_y, q=1-alpha)
    else:
        quantile_score_calib = quantile_corrected(score_calib, alpha=alpha)

    test_prediction = score_inv_fn(reg, quantile_score_calib, dataloader_test.X.copy()[:, obs_index])
    coverage = [test_prediction[0][n] <= y <= test_prediction[1][n] for n, y in enumerate(Y_test)]
    coverage = np.array(coverage)
    if reweight and estimation_reweight:
        return np.mean(coverage), calib_likelihood_ratio, true_calib_likelihood_ratio, errors
    return np.mean(coverage)


def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h

def quantile_corrected(x, alpha):
    n_x = len(x)
    if (1-alpha)*(1+1/n_x) > 1:
        return np.inf
    else:
        return np.quantile(x, (1-alpha)*(1+1/n_x), method='higher')

def weighted_quantile_matrix(X, W, q):
    X = np.asarray(X)
    W = np.asarray(W)

    if X.shape != W.shape:
        raise ValueError("X and W must have the same shape")

    quantiles = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        sorter = np.argsort(X[i])
        x_sorted = X[i][sorter]
        w_sorted = W[i][sorter]

        cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
        idx = np.searchsorted(cdf, q, side="right")
        quantiles[i] = x_sorted[idx]

    return quantiles

def predict_basemodel(fitted_basemodel, X_test, target='Mean', basemodel='Linear', alpha=0.1):

    assert target in ['Mean', 'Quantiles'], 'regression must be Mean or Quantiles.'
    assert basemodel in ['Linear', 'RF', 'NNet', 'XGBoost','histgbdt','NNet_missing'], 'regression must be Linear, RF or NNet.'

    if target == 'Mean':
        predictions = fitted_basemodel.predict(X_test)
    elif target == 'Quantiles':
        a_low = alpha/2
        a_high = 1-alpha/2
        if basemodel == 'Linear':
            predictions = {'y_inf': fitted_basemodel['q_low'].predict(X_test),
                           'y_sup': fitted_basemodel['q_high'].predict(X_test)}
        elif basemodel == 'RF':
            both_pred = fitted_basemodel.predict(X_test, quantiles=[a_low, a_high])
            predictions = {'y_inf': both_pred[:, 0],
                           'y_sup': both_pred[:, 1]}
        elif basemodel == 'XGBoost' or basemodel == 'histgbdt':
            predictions = {'y_inf': fitted_basemodel['q_low'].predict(X_test),
                           'y_sup': fitted_basemodel['q_high'].predict(X_test)}
        elif basemodel == 'NNet' or basemodel =='NNet_missing':
            both_pred = fitted_basemodel.predict(X_test)
            predictions = {'y_inf': both_pred[:, 0],
                           'y_sup': both_pred[:, 1]}
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return predictions

def fit_basemodel(X_train, Y_train, target='Mean', basemodel='Linear', alpha=0.1, params_basemodel={}):

    cores = params_basemodel['cores']

    if target == 'Mean':
        if basemodel == 'Linear':
            trained_model = LinearRegression(n_jobs=cores).fit(X_train,Y_train)

    elif target == 'Quantiles':
        a_low = alpha/2
        a_high = 1-alpha/2
        if basemodel == 'Linear':
            trained_model = {'q_low': QuantileRegressor(quantile=a_low, solver='highs', alpha=0).fit(X_train,Y_train),
                             'q_high': QuantileRegressor(quantile=a_high, solver='highs', alpha=0).fit(X_train,Y_train)}
        elif basemodel == 'histgbdt':
            trained_model = {'q_low': HistGradientBoostingRegressor(loss="quantile", quantile=a_low).fit(X_train,Y_train),
                             'q_high': HistGradientBoostingRegressor(loss="quantile", quantile=a_high).fit(X_train,Y_train)}

    return trained_model


class Miss_proba_simple_model:

    def __init__(self, data, model='xgboost', epoch=300, **hyperparam):
        self.data = data
        self.model_type = model
        self.hyperparam = hyperparam
        self.epoch = epoch
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.fetch_data(epoch=self.epoch)

        if model == 'xgboost':
            self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", **hyperparam)
        elif model == 'histgbdt':
            self.model = HistGradientBoostingClassifier(**hyperparam)
        elif model == 'logistic':
            self.model = LogisticRegression(**hyperparam)
        else:
            raise ValueError(f"Unsupported model type: {model}")

    def fetch_data(self, epoch=300):
        all_X, all_Y = [], []
        all_X_test, all_Y_test = [], []
        for i in range(epoch):
            for x_batch, y_batch in self.data:
                x_numpy = x_batch.numpy()
                y_numpy = y_batch.numpy()
                if i == epoch - 1:
                    all_X_test.append(x_numpy)
                    all_Y_test.append(y_numpy)
                else:
                    all_X.append(x_numpy)
                    all_Y.append(y_numpy)

        X_train = np.vstack(all_X)
        Y_train = np.vstack(all_Y).flatten()
        X_test = np.vstack(all_X_test)
        Y_test = np.vstack(all_Y_test).flatten()

        n_pos = np.sum(Y_train == 1)
        n_neg = np.sum(Y_train == 0)

        if n_pos > n_neg:
            X_neg = X_train[Y_train == 0]
            Y_neg = Y_train[Y_train == 0]
            X_neg_upsampled, Y_neg_upsampled = resample(X_neg, Y_neg, replace=True, n_samples=n_pos, random_state=42)

            X_train = np.vstack((X_train[Y_train == 1], X_neg_upsampled))
            Y_train = np.hstack((Y_train[Y_train == 1], Y_neg_upsampled))
        elif n_neg > n_pos:
            X_pos = X_train[Y_train == 1]
            Y_pos = Y_train[Y_train == 1]
            X_pos_upsampled, Y_pos_upsampled = resample(X_pos, Y_pos, replace=True, n_samples=n_neg, random_state=42)

            X_train = np.vstack((X_train[Y_train == 0], X_pos_upsampled))
            Y_train = np.hstack((Y_train[Y_train == 0], Y_pos_upsampled))
        else:
            X_train = X_train
            Y_train = Y_train

        return X_train, Y_train, X_test, Y_test

    def train(self):
        if self.model_type == 'logistic':
            train_mask = np.isnan(self.X_train).astype(int)
            x_train_imputed = self.imputer.fit_transform(self.X_train)
            self.X_train = np.hstack((x_train_imputed, train_mask))

            test_mask = np.isnan(self.X_test).astype(int)
            x_test_imputed = self.imputer.transform(self.X_test)
            self.X_test = np.hstack((x_test_imputed, test_mask))

        self.model.fit(self.X_train, self.Y_train)
        y_pred_test = self.model.predict_proba(self.X_test)[:, 1]
        try:
            self.loss = log_loss(self.Y_test, y_pred_test)
        except:
            self.loss = 0

    def normalize_eval(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = self.data.scaler.transform(x)
        if self.model_type == 'logistic':
            mask = np.isnan(x).astype(int)
            x_imputed = self.imputer.transform(x)
            x = np.hstack((x_imputed, mask))
        proba = self.model.predict_proba(x)[:, 1]
        normalize_proba = np.clip(proba / np.quantile(proba, 0.9),0,1)
        return normalize_proba

    def eval(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = self.data.scaler.transform(x)
        if self.model_type == 'logistic':
            mask = np.isnan(x).astype(int)
            x_imputed = self.imputer.transform(x)
            x = np.hstack((x_imputed, mask))
        proba = self.model.predict_proba(x)[:, 1]
        proba = np.clip(proba, 0, 0.9)
        return proba/(1-proba)

    def error(self, x, M, M_ref):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(M, torch.Tensor):
            M = M.numpy()
        if isinstance(M_ref, torch.Tensor):
            M_ref = M_ref.numpy()

        row_check = (M[:, M_ref == 0] == 1).any(axis=1)
        y = (~row_check).astype(int)

        x = self.data.scaler.transform(x)
        if self.model_type == 'logistic':
            mask = np.isnan(x).astype(int)
            x_imputed = self.imputer.transform(x)
            x = np.hstack((x_imputed, mask))

        proba = self.model.predict_proba(x)[:, 1]

        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(y, proba)
        errors = []
        for t in thresholds:
            y_pred = (proba >= t).astype(int)
            err = np.mean(y_pred != y)
            errors.append(err)

        min_error = np.min(errors)
        best_threshold = thresholds[np.argmin(errors)]
        return min_error

class gaussian_linear:
    def __init__(self, mu, Sigma, beta, noise_std=1, seed=5, n=1000):
        self.d = len(mu)
        self.mu = mu
        self.Sigma = Sigma
        self.beta = beta
        self.noise_std = noise_std
        np.random.seed(seed)
        self.X, self.Y = self.generate_data(n)
        self.patterns = list(itertools.product([0, 1], repeat=self.d))

    def generate_data(self, n):
        X = np.random.multivariate_normal(self.mu, self.Sigma, size=n)
        Y_reg = X.dot(self.beta)
        eps = np.random.normal(loc=0, scale=self.noise_std, size=n)
        Y = Y_reg + eps
        return X, Y

    def get_conditional_mis(self, x_obs, y, obs_idx):
        mis_idx = list(set(range(self.d)) - set(obs_idx))

        mu_obs = self.mu[obs_idx]
        mu_mis = self.mu[mis_idx]

        Sigma_oo = self.Sigma[np.ix_(obs_idx, obs_idx)]
        Sigma_om = self.Sigma[np.ix_(obs_idx, mis_idx)]
        Sigma_mo = self.Sigma[np.ix_(mis_idx, obs_idx)]
        Sigma_mm = self.Sigma[np.ix_(mis_idx, mis_idx)]

        beta_obs = self.beta[obs_idx]
        beta_mis = self.beta[mis_idx]

        mu_Y = self.beta @ self.mu
        Sigma_Y = self.beta @ self.Sigma @ self.beta + self.noise_std ** 2

        Sigma_oY = Sigma_oo @ beta_obs + Sigma_om @ beta_mis
        Sigma_mY = Sigma_mo @ beta_obs + Sigma_mm @ beta_mis

        mu_obsY = np.concatenate([mu_obs, [mu_Y]])
        Sigma_obsY = np.block([
            [Sigma_oo, Sigma_oY[:, None]],
            [Sigma_oY[None, :], Sigma_Y]
        ])

        Sigma_m_obsY = np.hstack([Sigma_mo, Sigma_mY[:, None]])

        obsY_concat = np.concatenate([x_obs, [y]])
        diff = obsY_concat - mu_obsY

        cond_mean = mu_mis + Sigma_m_obsY @ np.linalg.solve(Sigma_obsY, diff)
        cond_cov = Sigma_mm - Sigma_m_obsY @ np.linalg.solve(Sigma_obsY, Sigma_m_obsY.T)

        return cond_mean, cond_cov

    def generate_mcar(self, missing_prob=0.2):
        self.missing_prob = missing_prob
        mask = np.random.rand(*self.X.shape) < self.missing_prob
        self.X_missing = self.X.copy()
        self.X_missing[mask] = np.nan
        self.M = mask

    def all_imputation(self, mean_shift=-0.2, mean_scale=1.2, var_perturb=0.1):
        self.mean_shift = mean_shift
        self.mean_scale = mean_scale
        self.var_perturb = var_perturb

        patterns = list(itertools.product([0, 1], repeat=self.d))
        self.imputation_map = {}
        for pattern in patterns: # missing pattern
            obs_idx = [i for i, val in enumerate(pattern) if val == 0]
            mis_idx = [i for i in range(self.d) if i not in obs_idx]

            mu_obs = self.mu[obs_idx]
            mu_mis = self.mu[mis_idx]

            Sigma_oo = self.Sigma[np.ix_(obs_idx, obs_idx)]
            Sigma_om = self.Sigma[np.ix_(obs_idx, mis_idx)]
            Sigma_mo = self.Sigma[np.ix_(mis_idx, obs_idx)]
            Sigma_mm = self.Sigma[np.ix_(mis_idx, mis_idx)]

            beta_obs = self.beta[obs_idx]
            beta_mis = self.beta[mis_idx]

            mu_Y = self.beta @ self.mu
            Sigma_Y = self.beta @ self.Sigma @ self.beta + self.noise_std ** 2

            Sigma_oY = Sigma_oo @ beta_obs + Sigma_om @ beta_mis
            Sigma_mY = Sigma_mo @ beta_obs + Sigma_mm @ beta_mis

            cond_mean_mis = mu_mis
            cond_cov_mis = Sigma_mm

            perturbed_mean_mis = mean_scale * cond_mean_mis + mean_shift
            perturbed_cov_mis = cond_cov_mis + np.eye(len(mis_idx)) * var_perturb

            final_mean = np.zeros(self.d + 1)
            final_mean[obs_idx] = mu_obs
            final_mean[mis_idx] = perturbed_mean_mis
            final_mean[-1] = mu_Y

            final_cov = np.zeros((self.d + 1, self.d + 1))
            final_cov[np.ix_(obs_idx, obs_idx)] = Sigma_oo
            final_cov[np.ix_(obs_idx, mis_idx)] = Sigma_om
            final_cov[np.ix_(mis_idx, obs_idx)] = Sigma_mo
            final_cov[np.ix_(mis_idx, mis_idx)] = perturbed_cov_mis
            final_cov[-1, -1] = Sigma_Y

            final_cov[np.ix_(obs_idx, [-1])] = Sigma_oY[:, None]
            final_cov[np.ix_([-1], obs_idx)] = Sigma_oY[None, :]
            final_cov[np.ix_(mis_idx, [-1])] = Sigma_mY[:, None]
            final_cov[np.ix_([-1], mis_idx)] = Sigma_mY[None, :]

            self.imputation_map[pattern] = (final_mean, final_cov)

        return self.imputation_map

    def impute_with_perturbation(self):
        self.X_imputed = self.X_missing.copy()
        for i in range(len(self.X)):
            x_row = self.X_missing[i]
            y_val = self.Y[i]
            obs_idx = [j for j in range(self.d) if not np.isnan(x_row[j])]
            mis_idx = [j for j in range(self.d) if np.isnan(x_row[j])]

            if not mis_idx:
                continue

            x_obs = x_row[obs_idx]
            cond_mean, cond_cov = self.get_conditional_mis(x_obs, y_val, obs_idx)

            perturbation_mean = cond_mean * self.mean_scale + self.mean_shift
            perturbation_cov = cond_cov + np.eye(len(cond_cov)) * self.var_perturb

            imputed_values = np.random.multivariate_normal(perturbation_mean, perturbation_cov)

            self.X_imputed[i, mis_idx] = imputed_values

    def density_under_imputation_distribution(self, x, y):
        xy = np.concatenate([x, [y]])
        density = 0
        patterns = list(itertools.product([0, 1], repeat=self.d))
        mcar_prob = (self.missing_prob ** np.sum(patterns, axis=1)) * ((1-self.missing_prob) ** (self.d - np.sum(patterns, axis=1)))

        for pattern, prob in zip(patterns, mcar_prob):
            mean, cov = self.imputation_map[pattern]
            density += prob * multivariate_normal.pdf(xy, mean=mean, cov=cov)

        return density

    def get_joint_distribution(self):
        mu_Y = self.beta @ self.mu
        Sigma_Y = self.beta @ self.Sigma @ self.beta + self.noise_std ** 2
        Sigma_XY = self.Sigma @ self.beta
        Sigma_YX = Sigma_XY.T

        mu_joint = np.concatenate([self.mu, [mu_Y]])
        Sigma_joint = np.block([
            [self.Sigma, Sigma_XY[:, None]],
            [Sigma_YX[None, :], np.array([[Sigma_Y]])]
        ])
        return mu_joint, Sigma_joint

    def get_marginal_distribution(self, obs_idx):
        mu_joint, Sigma_joint = self.get_joint_distribution()
        mu_marginal = mu_joint[obs_idx]
        Sigma_marginal = Sigma_joint[np.ix_(obs_idx, obs_idx)]
        return mu_marginal, Sigma_marginal

    def get_imputed_marginal_distribution(self, obs_idx):
        imputed_mu_sigma = dict()
        for key in self.imputation_map:
            mu_joint, Sigma_joint = self.imputation_map[key]
            mu_marginal = mu_joint[obs_idx]
            Sigma_marginal = Sigma_joint[np.ix_(obs_idx, obs_idx)]
            imputed_mu_sigma[key] = (mu_marginal, Sigma_marginal)
        return imputed_mu_sigma

    def density_under_joint_distribution(self, x, y):
        mu_joint, Sigma_joint = self.get_joint_distribution()
        xy = np.concatenate([x, [y]])
        return multivariate_normal.pdf(xy, mean=mu_joint, cov=Sigma_joint)

    def marginal_density_under_imputation_distribution(self, x_obs, y, obs_idx):
        patterns = list(itertools.product([0, 1], repeat=self.d))
        density = 0
        mcar_prob = (self.missing_prob ** np.sum(patterns, axis=1)) * ((1-self.missing_prob) ** (self.d - np.sum(patterns, axis=1)))
        obs_y_concat = np.concatenate([x_obs, [y]])

        for pattern, prob in zip(patterns, mcar_prob):
            mean, cov = self.imputation_map[pattern]
            mean_obs_y = mean[np.ix_(obs_idx + [-1])]
            cov_obs_y = cov[np.ix_(obs_idx + [-1], obs_idx + [-1])]

            density += prob * multivariate_normal.pdf(obs_y_concat, mean=mean_obs_y, cov=cov_obs_y)

        return density

    def marginal_density_under_joint_distribution(self, x_obs, y, obs_idx, get_marginal_params=False):
        mu_joint, Sigma_joint = self.get_joint_distribution()
        idx = obs_idx + [-1]
        mu_marginal = mu_joint[idx]
        Sigma_marginal = Sigma_joint[np.ix_(idx, idx)]
        obs_y_concat = np.concatenate([x_obs, [y]])
        if get_marginal_params:
            return mu_marginal, Sigma_marginal
        return multivariate_normal.pdf(obs_y_concat, mean=mu_marginal, cov=Sigma_marginal)

    def likelihood_ratio_estimation(self, x_obs, y, obs_idx):
        return self.marginal_density_under_joint_distribution(x_obs, y, obs_idx) / self.marginal_density_under_imputation_distribution(x_obs, y, obs_idx)

    def estimate_lr_range(self):
        self.lr_range = defaultdict(lambda: [float('inf'), -float('inf')])
        for pattern in self.patterns:
            obs_idx = [i for i, val in enumerate(pattern) if val == 0]
            for i in range(len(self.X_imputed)):
                x,y = self.X_imputed[i], self.Y[i]
                lr = self.likelihood_ratio_estimation(x[obs_idx],y,obs_idx)
                self.lr_range[pattern][0] = min(lr, self.lr_range[pattern][0])
                self.lr_range[pattern][1] = max(lr, self.lr_range[pattern][1])
        return self.lr_range

class Data_for_mask:

    def __init__(self, X, M, numerical_feature=None, batch_size=128, test=0.2, contain_y=True, dataset=None, one_M=None):
        self.N = X.shape[0]
        self.d = X.shape[1]
        self.contain_y = contain_y
        if contain_y:
            self.d_x = X.shape[1] - 1
        else:
            self.d_x = X.shape[1]

        if numerical_feature is None:
            self.numerical_feature = list(range(self.d))
        else:
            self.numerical_feature = numerical_feature
        self.batch_size = batch_size
        self.test_n = int(self.N * test)
        self.M_original = M.copy()
        self.X = X.copy()
        self.M_num = np.array([[True,]*i + [False,]*(self.d_x-i) for i in range(self.d_x)])

        self.scaler = StandardScaler()
        self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)

        self.X_test = self.X[:self.test_n]
        self.X = self.X[self.test_n:]

        self.X_test_torch = torch.tensor(self.X_test, dtype=torch.float32)
        self.X_torch = torch.tensor(self.X, dtype=torch.float32)
        self.N_train = self.X.shape[0]

        self.one_M = one_M

        return


    def __len__(self):
        return len(self.X_torch)

    def __iter__(self):
        self.start_idx = 0
        self.indices = torch.arange(len(self.X_torch))
        # weights = np.linspace(1, 4, self.d_x)
        # weights = weights / weights.sum()
        # indices = np.random.choice(self.d_x, size=self.N, replace=True,  p=weights)
        if self.one_M is None:
            indices = np.random.choice(self.d_x, size=self.N, replace=True)
            self.M = self.M_num[indices]
            for row in self.M:
                np.random.shuffle(row)
        else:
            self.M = np.tile(self.one_M, (self.N, 1))

        assert self.M.shape == self.M_original.shape
        row_check = (self.M & self.M_original).sum(axis=1) == self.M_original.sum(axis=1)
        self.Y = row_check.astype(int)

        if self.contain_y:
            self.M = np.hstack((self.M, np.zeros((self.M.shape[0], 1))))

        self.M_test = self.M[:self.test_n]
        self.Y_test = self.Y[:self.test_n]

        self.M = self.M[self.test_n:]
        self.Y = self.Y[self.test_n:]

        self.M_torch = torch.tensor(self.M, dtype=torch.bool)
        self.Y_torch = torch.tensor(self.Y, dtype=torch.float32).view(-1, 1)
        self.X_nan_torch = self.X_torch.masked_fill(self.M_torch, float('nan'))

        self.M_test_torch = torch.tensor(self.M_test, dtype=torch.bool)
        self.Y_test_torch = torch.tensor(self.Y_test, dtype=torch.float32).view(-1, 1)
        self.X_nan_test_torch = self.X_test_torch.masked_fill(self.M_test_torch, float('nan'))

        return self

    def __next__(self):
        if self.start_idx < self.N_train:
            end_idx = min(self.start_idx + self.batch_size, self.N_train)
            batch_indices = self.indices[self.start_idx:end_idx]
            self.start_idx = end_idx
            return (self.X_nan_torch[batch_indices],
                    self.Y_torch[batch_indices])
        else:
            raise StopIteration