import os
import csv
from typing import Any, Dict, List, Optional

NA = "NA"

# Per-batch CSV schema. The first columns are run context; the rest are the
# metrics requested by the user. Prior-related columns stay "NA" for methods
# that do not use a class-prior (e.g. CLIPTTA baseline, source/zero-shot).
BATCH_COLUMNS: List[str] = [
    "seed", "dataset", "shift_type", "severity", "adaptation",
    "batch_idx", "batch_acc", "running_acc",
    "loss_total", "loss_scont", "loss_reg",
    "mean_confidence", "pred_entropy", "pred_entropy_norm",
    "prior_entropy", "prior_entropy_norm",
    "top1_prior_class", "top1_prior_value", "prior_delta_l1",
    "selected_ratio", "num_selected",
]

# One row per run (i.e. per seed / shift_type / severity).
FINAL_COLUMNS: List[str] = [
    "exp_name", "dataset", "shift_type", "severity", "seed",
    "adaptation", "base_model_name", "batch_size", "steps", "lr",
    "final_accuracy", "mean_batch_accuracy", "mean_confidence",
    "mean_pred_entropy_norm", "mean_prior_entropy_norm",
    "final_prior_entropy_norm", "final_top1_prior_value", "mean_selected_ratio",
]

# Keys that are aggregated (mean over batches) for the final summary.
_MEAN_KEYS = {
    "mean_batch_accuracy": "batch_acc",
    "mean_confidence": "mean_confidence",
    "mean_pred_entropy_norm": "pred_entropy_norm",
    "mean_prior_entropy_norm": "prior_entropy_norm",
    "mean_selected_ratio": "selected_ratio",
}
# Keys taken from the last batch with a valid value for the final summary.
_LAST_KEYS = {
    "final_prior_entropy_norm": "prior_entropy_norm",
    "final_top1_prior_value": "top1_prior_value",
}


def _fmt(value: Any) -> Any:
    """Format a cell: None -> NA, floats rounded for readability."""
    if value is None:
        return NA
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 6)
    return value


class MetricsLogger:
    """Writes per-batch and per-run metrics to CSV files under exp_dir.

    Usage:
        logger = MetricsLogger(exp_dir, static_meta)
        logger.start_run(seed, dataset, shift_type, severity)
        ... logger.log_batch(stats) per batch ...
        logger.end_run(final_accuracy)
        logger.close()
    """

    def __init__(self, exp_dir: str, static_meta: Dict[str, Any]) -> None:
        os.makedirs(exp_dir, exist_ok=True)
        self.exp_dir = exp_dir
        self.static_meta = static_meta  # exp_name, adaptation, base_model_name, batch_size, steps, lr

        self.batch_path = os.path.join(exp_dir, "batch_metrics.csv")
        self.final_path = os.path.join(exp_dir, "final_metrics.csv")

        self._batch_file = open(self.batch_path, "w", newline="")
        self._batch_writer = csv.DictWriter(
            self._batch_file, fieldnames=BATCH_COLUMNS, restval=NA, extrasaction="ignore"
        )
        self._batch_writer.writeheader()
        self._batch_file.flush()

        self._final_file = open(self.final_path, "w", newline="")
        self._final_writer = csv.DictWriter(
            self._final_file, fieldnames=FINAL_COLUMNS, restval=NA, extrasaction="ignore"
        )
        self._final_writer.writeheader()
        self._final_file.flush()

        self._run: Optional[Dict[str, Any]] = None

    def start_run(self, seed: int, dataset: str, shift_type: str, severity: int) -> None:
        self._run = {
            "seed": seed,
            "dataset": dataset,
            "shift_type": shift_type,
            "severity": severity,
            "rows": [],  # raw stats dicts (with None for missing values)
        }

    def log_batch(self, stats: Dict[str, Any]) -> None:
        if self._run is None:
            raise RuntimeError("MetricsLogger.log_batch called before start_run")

        ctx = {
            "seed": self._run["seed"],
            "dataset": self._run["dataset"],
            "shift_type": self._run["shift_type"],
            "severity": self._run["severity"],
            "adaptation": self.static_meta.get("adaptation"),
        }
        row = {**ctx, **stats}
        self._batch_writer.writerow({k: _fmt(row.get(k)) for k in BATCH_COLUMNS})
        self._batch_file.flush()
        self._run["rows"].append(stats)

    def end_run(self, final_accuracy: float) -> None:
        if self._run is None:
            raise RuntimeError("MetricsLogger.end_run called before start_run")
        rows = self._run["rows"]

        def _mean(key: str) -> Optional[float]:
            vals = [r.get(key) for r in rows if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        def _last(key: str) -> Optional[Any]:
            for r in reversed(rows):
                if r.get(key) is not None:
                    return r.get(key)
            return None

        final_row: Dict[str, Any] = {
            "exp_name": self.static_meta.get("exp_name"),
            "dataset": self._run["dataset"],
            "shift_type": self._run["shift_type"],
            "severity": self._run["severity"],
            "seed": self._run["seed"],
            "adaptation": self.static_meta.get("adaptation"),
            "base_model_name": self.static_meta.get("base_model_name"),
            "batch_size": self.static_meta.get("batch_size"),
            "steps": self.static_meta.get("steps"),
            "lr": self.static_meta.get("lr"),
            "final_accuracy": final_accuracy,
        }
        for out_key, src_key in _MEAN_KEYS.items():
            final_row[out_key] = _mean(src_key)
        for out_key, src_key in _LAST_KEYS.items():
            final_row[out_key] = _last(src_key)

        self._final_writer.writerow({k: _fmt(final_row.get(k)) for k in FINAL_COLUMNS})
        self._final_file.flush()
        self._run = None

    def close(self) -> None:
        try:
            self._batch_file.close()
        finally:
            self._final_file.close()
