"""BALD selection for categorical observations in NIMO candidate CSVs."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.special import xlogy
from .bayesian_classification import RBFClassifier


def bald_scores(probabilities):
    """Mutual information from posterior predictive probabilities [S, N, K]."""
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim != 3 or min(p.shape) < 1 or not np.isfinite(p).all()
            or (p < 0).any() or not np.allclose(p.sum(-1), 1)):
        raise ValueError('Expected normalized [posterior_sample, candidate, class] probabilities')
    mean = p.mean(0)
    return np.maximum(-np.sum(xlogy(mean, mean), axis=-1)
                      + np.mean(np.sum(xlogy(p, p), axis=-1), axis=0), 0)


class BALD:
    """Select a top-k BALD batch, without within-batch posterior updates.

    Exactly one rightmost objective column is used as a string class label.
    Blank cells are unobserved. Other columns must be finite numeric features.
    """
    def __init__(self, input_file, output_file, num_objectives=1, num_proposals=1,
                 samples=4096, lengthscale=0.25, prior_sd=2.0,
                 num_centers=21, scaling=None, seed=0, log_file=None):
        if num_objectives != 1:
            raise ValueError('BALD supports exactly one categorical objective column')
        if type(num_proposals) is not int or num_proposals < 1:
            raise ValueError('num_proposals must be a positive integer')
        if type(samples) is not int or samples < 2:
            raise ValueError('samples must be an integer >= 2')
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError('seed must be a nonnegative integer or None')
        self.input_file, self.output_file = Path(input_file), Path(output_file)
        self.log_file = Path(log_file) if log_file is not None else Path(str(output_file) + '.bald.json')
        paths = [self.input_file.resolve(), self.output_file.resolve(), self.log_file.resolve()]
        if len(set(paths)) != 3:
            raise ValueError('Input, output, and BALD log paths must differ')
        self.num_proposals, self.samples, self.seed = num_proposals, samples, seed
        self.classifier = RBFClassifier(lengthscale, prior_sd, num_centers, scaling)

    def load_data(self):
        with self.input_file.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            self.headers = next(reader, [])
            rows = list(reader)
        if len(self.headers) < 2 or len(set(self.headers)) != len(self.headers):
            raise ValueError('CSV needs distinct feature headers and one label header')
        if not rows or any(len(r) != len(self.headers) for r in rows):
            raise ValueError('CSV must contain nonempty rectangular data')
        try:
            x = np.asarray([[float(v) for v in r[:-1]] for r in rows])
        except ValueError as exc:
            raise ValueError('All feature columns must be numeric') from exc
        if not np.isfinite(x).all():
            raise ValueError('Features must not contain NaN or infinity')
        labels = [r[-1].strip() for r in rows]
        observed = np.array([i for i, label in enumerate(labels) if label], dtype=int)
        available = np.array([i for i, label in enumerate(labels) if not label], dtype=int)
        if self.num_proposals > len(available):
            raise ValueError('num_proposals exceeds the number of unobserved candidates')
        self.classes = sorted({labels[i] for i in observed})
        mapping = {label:i for i, label in enumerate(self.classes)}
        y = np.array([mapping[label] if label else -1 for label in labels], dtype=int)
        self.raw_rows = rows
        return x, y, observed, available

    def select(self):
        print("Start selection of proposals by BALD!")
        x, y, observed, available = self.load_data()
        # Validate the feature configuration even if no classifier can yet be fitted.
        self.classifier.features(x)
        scores = None
        if len(self.classes) < 2:
            ids = np.random.default_rng(self.seed).choice(available, self.num_proposals, replace=False).tolist()
            metadata = dict(self.classifier.feature_metadata, fallback='fewer_than_two_observed_classes_seeded_random')
        else:
            p, metadata = self.classifier.posterior_predictions(x, y, self.samples, self.seed)
            scores = bald_scores(p)[available]
            ids = available[np.lexsort((available, -scores))[:self.num_proposals]].tolist()
        if len(ids) != len(set(ids)) or not set(ids) <= set(available.tolist()):
            raise RuntimeError('Invalid BALD batch')
        metadata.update(method='BALD', seed=self.seed, samples=self.samples,
            classes=self.classes, n_observed=len(observed), n_candidates=len(x),
            num_proposals=self.num_proposals, batch_rule='top_k_single_query_scores',
            candidate_ids=available.tolist(), scores=None if scores is None else scores.tolist(), selected_ids=ids)
        # NIMO action IDs are zero-based CSV data-row indices, not sorted composition IDs.
        with self.output_file.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['actions'] + self.headers[:-1])
            writer.writerows([[i] + self.raw_rows[i][:-1] for i in ids])
        self.log_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        print("Proposals")
        for number, action in enumerate(ids, start=1):
            print("###")
            print("number =", number)
            print("actions = ", action)
            print("proposal = ", x[action])
            print("###")
        print("Finish selection of proposals!")
        return 'True'
