import numpy as np
from scipy.stats import ks_2samp

def psi_calculate(expected, actual, buckets=10):
    """Compute Population Stability Index (PSI)."""
    def scale_range(series, buckets):
        return np.interp(series, (series.min(), series.max()), (0, buckets - 0.001))

    expected_scaled = scale_range(expected, buckets)
    actual_scaled = scale_range(actual, buckets)

    expected_counts = np.histogram(expected_scaled, bins=buckets)[0]
    actual_counts = np.histogram(actual_scaled, bins=buckets)[0]

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return psi


def compute_drift_metrics(baseline, recent, column):
    """Return PSI, KS p-value, mean/std deltas."""
    b = baseline[column].dropna()
    r = recent[column].dropna()

    if len(b) < 20 or len(r) < 20:
        return {
            "PSI": None,
            "KS_pvalue": None,
            "mean_delta": None,
            "std_delta": None,
            "note": "Not enough data for drift analysis",
        }

    psi = psi_calculate(b, r)
    ks_p = ks_2samp(b, r).pvalue

    return {
        "PSI": psi,
        "KS_pvalue": ks_p,
        "mean_delta": r.mean() - b.mean(),
        "std_delta": r.std() - b.std(),
        "note": None,
    }

def psi_severity(psi: float | None) -> str:
    """Classify PSI into severity levels."""
    if psi is None:
        return "insufficient_data"
    if psi < 0.1:
        return "no_drift"
    if psi < 0.25:
        return "moderate_drift"
    return "significant_drift"
