"""Level set estimation using PHYSBO's latent Gaussian-process predictions.

Equations (9)--(12) of Ishibashi et al. (2025), arXiv:2503.20272.
Reference implementation: https://github.com/hideaki-ishibashi/stopping_LSE
This implementation uses epsilon as the FULL margin width (the reference
code's ``margin`` is epsilon / 2). An omitted epsilon uses Appendix B.1 with L=5 and confidence=0.99.
Batches use predicted-mean fantasies with fixed fitted GP parameters.
GP learning selects the best marginal likelihood from five fixed lengthscale starts.
This module selects and classifies only; margin confidence is not a stop decision.
"""
import csv
from pathlib import Path

import numpy as np
import physbo
from scipy.special import ndtr, ndtri


def _finite_scalar(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError('%s must be a finite number.' % name)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError('%s must be a finite number.' % name) from None
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError('%s must be finite%s.' % (name, ' and positive' if positive else ''))
    return value


def lse_statistics(mean, variance, threshold, epsilon):
    """Return marginal probabilities, labels and Eq. (9) acquisition values.

    Latent-function variance is required, not noisy observation variance.
    Labels are ``above``, ``below`` and ``margin``. The latter means the
    value is believed to be in (threshold-epsilon/2, threshold+epsilon/2],
    not that a new phase/category has been discovered.
    """
    threshold = _finite_scalar(threshold, 'lse_threshold')
    epsilon = _finite_scalar(epsilon, 'lse_epsilon', positive=True)
    mean = np.asarray(mean, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if mean.ndim != 1 or variance.shape != mean.shape:
        raise ValueError('Mean and variance must be matching one-dimensional arrays.')
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
        raise ValueError('GP predictions must be finite.')
    if np.any(variance < 0):
        raise ValueError('Latent variance must be nonnegative.')
    half = epsilon / 2
    std = np.sqrt(variance)
    above = np.empty_like(mean)
    below = np.empty_like(mean)
    margin = np.empty_like(mean)
    outside = np.empty_like(mean)
    stochastic = std > 0
    z = (mean[stochastic] - threshold) / std[stochastic]
    above[stochastic] = ndtr(z)
    below[stochastic] = ndtr(-z)
    lo = (threshold - half - mean[stochastic]) / std[stochastic]
    hi = (threshold + half - mean[stochastic]) / std[stochastic]
    # Use survival functions on the positive tail to avoid cancellation.
    mass = np.where(lo > 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
    margin[stochastic] = np.clip(mass, 0, 1)
    outside[stochastic] = np.clip(ndtr(lo) + ndtr(-hi), 0, 1)
    deterministic = ~stochastic
    above[deterministic] = mean[deterministic] > threshold
    below[deterministic] = mean[deterministic] <= threshold
    margin[deterministic] = ((mean[deterministic] > threshold - half) &
                             (mean[deterministic] <= threshold + half))
    outside[deterministic] = 1 - margin[deterministic]
    probabilities = np.column_stack((above, below, margin))
    # Match the reference implementation's argmax tie order: above, below, margin.
    labels = np.array(['above', 'below', 'margin'])[np.argmax(probabilities, axis=1)]
    acquisition = np.minimum(np.minimum(above, below), outside)
    return dict(prob_above=above, prob_below=below, prob_margin=margin,
                classification=labels, acquisition=acquisition)


# Appendix B.1 / the paper's test-function settings. Explicit epsilon overrides these.
DEFAULT_MARGIN_REPEATS = 5
DEFAULT_MARGIN_CONFIDENCE = 0.99

# Deterministic learning starts in standardized input units.
DEFAULT_GP_START_WIDTHS = (1.0, 0.1, 0.3, 3.0, 10.0)


def automatic_lse_epsilon(signal_variance, noise_variance, num_candidates):
    """Full margin width from Appendix B.1, Eq. (57), for a stationary kernel.

    Variances must use the same output units. L=5 and confidence=0.99 are
    margin-setting defaults, not a stopping rule or a guarantee of GP calibration.
    Candidate count covers the full fixed pool, including observed candidates.
    """
    signal = _finite_scalar(signal_variance, 'signal_variance', positive=True)
    noise = _finite_scalar(noise_variance, 'noise_variance', positive=True)
    if (isinstance(num_candidates, (bool, np.bool_)) or
            not isinstance(num_candidates, (int, np.integer)) or num_candidates < 1):
        raise ValueError('num_candidates must be a positive integer.')
    desired_variance = 1.0 / (1.0 / signal + DEFAULT_MARGIN_REPEATS / noise)
    # Inverse survival function, avoiding subtraction from one in the upper tail.
    z = -ndtri((1.0 - DEFAULT_MARGIN_CONFIDENCE) / (2 * int(num_candidates)))
    return _finite_scalar(2 * np.sqrt(desired_variance) * z, 'automatic epsilon', positive=True)


class LSE():
    """Single-objective, fixed-threshold LSE over a finite candidate CSV.

    A row contains numeric features followed by one numeric objective; an
    empty objective is unmeasured. At least two measured rows are needed.
    Omitted epsilon uses the paper's automatic width (L=5, confidence=0.99).
    GP hyperparameters, including observation noise, are learned from five
    deterministic lengthscale starts and selected by marginal likelihood.
    Each call fits on real observations only. Batches temporarily condition on
    predicted means with fixed hyperparameters; fantasies never enter the CSV
    or the reported real-data predictions. No observed row is proposed again.
    No previous model or state file is read. Selection and
    classification do not decide when an exploration loop should terminate.
    """

    def __init__(self, input_file, output_file=None, num_objectives=1,
                 num_proposals=1, lse_threshold=None, lse_epsilon=None,
                 output_res=False):
        if num_objectives != 1:
            raise ValueError('LSE supports num_objectives=1 only.')
        if (isinstance(num_proposals, (bool, np.bool_)) or
                not isinstance(num_proposals, (int, np.integer)) or num_proposals < 1):
            raise ValueError('num_proposals must be a positive integer.')
        self.num_proposals = int(num_proposals)
        self.threshold = _finite_scalar(lse_threshold, 'lse_threshold')
        self._manual_epsilon = (None if lse_epsilon is None else
                                _finite_scalar(lse_epsilon, 'lse_epsilon', positive=True))
        self.epsilon = self._manual_epsilon
        self.input_file = Path(input_file)
        self.output_file = Path(output_file) if output_file is not None else None
        if self.output_file is not None and self.output_file.resolve() == self.input_file.resolve():
            raise ValueError('Input and output files must be different.')
        self.output_res = output_res

    def load_data(self):
        """Read the NIMO numeric CSV while preserving row-based action IDs."""
        with self.input_file.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            try:
                self.header = next(reader)
            except StopIteration:
                raise ValueError('LSE candidates CSV is empty.') from None
            if len(self.header) < 2:
                raise ValueError('LSE needs at least one feature and one objective column.')
            rows = []
            for line, row in enumerate(reader, start=2):
                if len(row) != len(self.header):
                    raise ValueError('LSE CSV row %d has an unexpected number of columns.' % line)
                try:
                    features = [float(v) for v in row[:-1]]
                    objective = np.nan if not row[-1].strip() else float(row[-1])
                except ValueError:
                    raise ValueError('LSE CSV row %d must contain numeric values.' % line) from None
                if not np.all(np.isfinite(features)) or (row[-1].strip() and not np.isfinite(objective)):
                    raise ValueError('LSE CSV row %d has nonfinite values; use an empty objective for unmeasured rows.' % line)
                rows.append(features + [objective])
        if not rows:
            raise ValueError('LSE candidates CSV has no candidate rows.')
        values = np.asarray(rows, dtype=float)
        self.features = values[:, :-1]
        self.objectives = values[:, -1]
        self.observed = np.flatnonzero(~np.isnan(self.objectives))
        self.unmeasured = np.flatnonzero(np.isnan(self.objectives))
        return self.features, self.objectives, self.observed, self.unmeasured

    def _predict(self):
        """Fit deterministic exact PHYSBO GP and return latent moments in original units."""
        if len(self.observed) < 2:
            raise ValueError('LSE needs at least two measured rows; initialize using RE or DOE.')
        x_scale = np.std(self.features, axis=0)
        x_scale[x_scale == 0] = 1
        x = (self.features - np.mean(self.features, axis=0)) / x_scale
        y = self.objectives[self.observed]
        offset = float(np.mean(y))
        scale = float(np.std(y))
        self.hyperparameters_fitted = bool(scale > np.finfo(float).eps * max(1.0, abs(offset)))
        if not self.hyperparameters_fitted:
            # Constant initial observations cannot identify output variation.
            # Keep a nonzero prior instead of fitting the signal variance to zero.
            scale = max(1.0, abs(offset))
        y = (y - offset) / scale
        # Bound normalized scales so optimizer trial covariances remain well conditioned.
        gp = physbo.gp.Model(lik=physbo.gp.lik.Gauss(min_params=1e-4, max_params=10.0),
            mean=physbo.gp.mean.Const(0.0),
            cov=physbo.gp.cov.Gauss(x.shape[1], width=1.0, scale=1.0, ard=True,
                min_width=1e-3, max_width=1e3, min_scale=1e-3, max_scale=10.0))
        if self.hyperparameters_fitted:
            config = physbo.misc.SetConfig()
            # Each deterministic start runs full-data L-BFGS-B without RNG mutation.
            config.learning.method = 'bfgs'
            config.learning.num_init_params_search = 0
            config.learning.batch_size = None
            config.learning.max_iter = 200
            config.learning.is_disp = False
            self._fit_gp(gp, x[self.observed], y, config)
        self._gp = gp
        self._x = x
        self._train_x = x[self.observed].copy()
        self._train_y = y.copy()
        self._offset, self._scale = offset, scale
        signal = float(np.asarray(gp.prior.cov.get_cov(x[:1], diag=True)).reshape(-1)[0])
        noise = float(np.asarray(gp.lik.get_cov(1)).reshape(-1)[0])
        self.signal_variance = signal * scale**2
        self.noise_variance = noise * scale**2
        self._automatic_epsilon = scale * automatic_lse_epsilon(signal, noise, len(x))
        gp.prepare(self._train_x, self._train_y)
        return self._posterior_moments()

    def _fit_gp(self, gp, x, y, config):
        """Keep the best finite marginal-likelihood fit from five fixed starts."""
        initial = gp.params.copy()
        # Gauss covariance stores log widths before its final log signal SD.
        width_start = gp.lik.num_params + gp.prior.mean.num_params
        width_end = width_start + gp.prior.cov.num_params - 1
        best_score, best_params = np.inf, None
        for width in DEFAULT_GP_START_WIDTHS:
            params = initial.copy()
            params[width_start:width_end] = np.log(width)
            # Reset noise, mean and signal as well as widths before every fit.
            gp.set_params(params)
            gp.fit(x, y, config)
            score = float(gp.eval_marlik(gp.params, x, y))
            if np.isfinite(score) and score < best_score:
                best_score, best_params = score, gp.params.copy()
        if best_params is None:
            raise ValueError('LSE GP learning returned no finite marginal likelihood.')
        gp.set_params(best_params)

    def _posterior_moments(self):
        """Current latent moments; hyperparameters are not refitted here."""
        mean = np.asarray(self._gp.get_post_fmean(self._train_x, self._x)).reshape(-1)
        variance = np.asarray(self._gp.get_post_fcov(self._train_x, self._x)).reshape(-1)
        if not np.all(np.isfinite(variance)) or np.any(variance < -1e-8):
            raise ValueError('PHYSBO returned invalid latent variances.')
        return (self._offset + self._scale * mean,
                self._scale**2 * np.maximum(variance, 0))

    def _condition_on_prediction(self, action, predicted_mean):
        """Temporarily add one noisy-observation fantasy at its predicted mean."""
        self._train_x = np.vstack((self._train_x, self._x[action]))
        value = (predicted_mean - self._offset) / self._scale
        self._train_y = np.append(self._train_y, value)
        self._gp.prepare(self._train_x, self._train_y)

    def evaluate(self):
        """Compute classification and acquisition values from the current CSV only."""
        self.load_data()
        mean, variance = self._predict()
        self.epsilon = (self._automatic_epsilon if self._manual_epsilon is None
                        else self._manual_epsilon)
        stats = lse_statistics(mean, variance, self.threshold, self.epsilon)
        self.result = dict(stats, mean=mean, variance=variance,
            observed=self.observed.copy(), unmeasured=self.unmeasured.copy(),
            hyperparameters_fitted=self.hyperparameters_fitted,
            epsilon=self.epsilon,
            epsilon_source='auto' if self._manual_epsilon is None else 'manual',
            margin_repeats=DEFAULT_MARGIN_REPEATS if self._manual_epsilon is None else None,
            margin_confidence=DEFAULT_MARGIN_CONFIDENCE if self._manual_epsilon is None else None,
            signal_variance=getattr(self, 'signal_variance', None),
            noise_variance=getattr(self, 'noise_variance', None))
        return self.result

    def _write_results(self, result):
        """Write predictions for all candidates without overwriting the input."""
        if not self.output_res:
            return
        path = Path('output_res.csv') if self.output_res is True else Path(self.output_res)
        protected = [self.input_file.resolve()]
        if self.output_file is not None:
            protected.append(self.output_file.resolve())
        if path.resolve() in protected:
            raise ValueError('LSE diagnostics must not overwrite candidates or proposals.')
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['actions'] + self.header[:-1] +
                ['observed', 'mean', 'variance', 'prob_above', 'prob_below',
                 'prob_margin', 'classification', 'acquisition'] +
                ['lse_epsilon', 'epsilon_source', 'selected_rank', 'batch_acquisition'])
            selected = {step['action']: step for step in result.get('batch_trace', [])}
            for i, features in enumerate(self.features):
                writer.writerow([i] + features.tolist() + [bool(not np.isnan(self.objectives[i]))] +
                    [result[key][i] for key in ('mean', 'variance', 'prob_above',
                     'prob_below', 'prob_margin', 'classification', 'acquisition')] +
                    [result['epsilon'], result['epsilon_source'],
                     selected.get(i, {}).get('step', ''),
                     selected.get(i, {}).get('acquisition', '')])

    def select(self):
        """Write a no-repeat batch using fixed-parameter predicted-mean fantasies.

        The original GP is restored before return. result/diagnostic moments
        describe real observations only; batch_trace holds conditional scores.
        Automatic epsilon is held fixed within this batch and recomputed from
        current real data at the next call. All score ties use CSV row order.
        """
        if self.output_file is None:
            raise ValueError('LSE selection requires output_file.')
        result = self.evaluate()
        if len(self.unmeasured) == 0:
            raise ValueError('LSE has no unmeasured candidates.')
        if self.num_proposals > len(self.unmeasured):
            raise ValueError('num_proposals exceeds the unmeasured candidate count.')
        remaining = self.unmeasured.tolist()
        mean, variance, stats = result['mean'], result['variance'], result
        self.actions, self.batch_trace = [], []
        # One-point selection retains the original rule without any conditioning.
        real_x, real_y = self._train_x, self._train_y
        try:
            for step in range(self.num_proposals):
                action = int(remaining[np.argmax(stats['acquisition'][remaining])])
                self.actions.append(action)
                self.batch_trace.append(dict(step=step + 1, action=action,
                    predicted_mean=float(mean[action]), latent_variance=float(variance[action]),
                    acquisition=float(stats['acquisition'][action])))
                remaining.remove(action)
                if step + 1 < self.num_proposals:
                    self._condition_on_prediction(action, mean[action])
                    mean, variance = self._posterior_moments()
                    stats = lse_statistics(mean, variance, self.threshold, self.epsilon)
        finally:
            if self.num_proposals > 1:
                self._train_x, self._train_y = real_x, real_y
                self._gp.prepare(real_x, real_y)
        result.update(selected_actions=self.actions.copy(), batch_trace=list(self.batch_trace))
        self._write_results(result)
        with self.output_file.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['actions'] + self.header[:-1])
            writer.writerows([action] + self.features[action].tolist() for action in self.actions)
        print('LSE proposals: actions = %s' % self.actions)
        return 'True'
