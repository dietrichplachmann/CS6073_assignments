"""Local zero-dimensional persistence features for tabular county data.

For each county, form a point cloud from that county and its nearest training
counties in standardized predictor space. In a Vietoris-Rips filtration, the
finite H0 death times equal the minimum spanning tree edge lengths. Summaries
of those lengths become extra numeric inputs to a neural network.
"""

import numpy as np


FEATURE_NAMES = ("h0_q25", "h0_median", "h0_q75", "h0_max", "h0_std")


def _h0_death_times(cloud):
    """Compute H0 persistence death times as Euclidean MST edge lengths."""
    distances = np.sqrt(np.maximum(
        np.sum((cloud[:, None, :] - cloud[None, :, :]) ** 2, axis=2), 0.0,
    ))
    visited = np.zeros(len(cloud), dtype=bool)
    visited[0] = True
    nearest = distances[0].copy()
    deaths = np.empty(len(cloud) - 1, dtype=np.float64)
    for edge in range(len(deaths)):
        nearest[visited] = np.inf
        next_point = int(np.argmin(nearest))
        deaths[edge] = nearest[next_point]
        visited[next_point] = True
        nearest = np.minimum(nearest, distances[next_point])
    return deaths


class LocalPersistenceFeatures:
    """Fit neighborhoods and feature scaling on the training split only."""

    def __init__(self, neighbors=12):
        self.neighbors = neighbors

    def _summaries(self, queries, exclude_self):
        if queries.shape[1] != self.reference.shape[1]:
            raise ValueError("Topological inputs have the wrong feature count")
        if exclude_self and len(queries) != len(self.reference):
            raise ValueError("Self-exclusion requires the training rows in order")
        distances_sq = np.maximum(
            np.sum(queries * queries, axis=1)[:, None]
            + np.sum(self.reference * self.reference, axis=1)[None, :]
            - 2.0 * queries @ self.reference.T,
            0.0,
        )
        if exclude_self:
            np.fill_diagonal(distances_sq, np.inf)
        summaries = np.empty((len(queries), len(FEATURE_NAMES)), dtype=np.float64)
        for row, query in enumerate(queries):
            neighbors = np.argpartition(
                distances_sq[row], self.neighbors - 1,
            )[:self.neighbors]
            cloud = np.vstack((query, self.reference[neighbors]))
            deaths = _h0_death_times(cloud)
            summaries[row] = (
                *np.quantile(deaths, (0.25, 0.50, 0.75)),
                deaths.max(), deaths.std(),
            )
        return summaries

    def fit_transform(self, training_features):
        if self.neighbors < 2 or self.neighbors >= len(training_features):
            raise ValueError("neighbors must be at least 2 and below training size")
        if not np.isfinite(training_features).all():
            raise ValueError("Topological inputs must be finite")
        self.reference = np.asarray(training_features, dtype=np.float64).copy()
        raw = self._summaries(self.reference, exclude_self=True)
        self.means = raw.mean(axis=0)
        self.scales = raw.std(axis=0)
        self.scales[self.scales == 0] = 1.0
        return (raw - self.means) / self.scales

    def transform(self, features):
        if not np.isfinite(features).all():
            raise ValueError("Topological inputs must be finite")
        raw = self._summaries(np.asarray(features, dtype=np.float64), exclude_self=False)
        return (raw - self.means) / self.scales

    def save(self, path):
        np.savez(
            path, reference=self.reference, neighbors=self.neighbors,
            means=self.means, scales=self.scales,
            feature_names=np.array(FEATURE_NAMES),
        )
