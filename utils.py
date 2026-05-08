"""
Copyright HallResearch.ai 2026

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
import tensorflow_lattice as tfl
import tf_keras as keras


@dataclass
class TrainedModel:
    """Stores a trained model and its evaluation artifacts.

    Attributes:
        name: Display name for the model variant.
        model: Trained Keras model.
        calibrators: Mapping from feature name to calibrator layer.
        feature_cols: Ordered modeling columns used by the model.
        monotonicity_map: Mapping from feature name to monotonicity sign.
        keypoints_map: Mapping from feature name to calibration keypoints.
        history: Keras training history.
        valid_scores: Validation-set predicted probabilities.
        cutoff: Validation-selected score cutoff.
        valid_auc: Validation AUC.
    """

    name: str
    model: keras.Model
    calibrators: dict
    feature_cols: list
    monotonicity_map: dict
    keypoints_map: dict
    history: keras.callbacks.History
    valid_scores: np.ndarray
    cutoff: float
    valid_auc: float


def frame_to_inputs(frame, columns):
    """Converts a dataframe slice into a Keras input dictionary.

    Args:
        frame: Input dataframe.
        columns: Ordered feature columns to extract.

    Returns:
        Dict[str, np.ndarray]: Column-wise float32 arrays.
    """
    return {col: frame[[col]].astype("float32").to_numpy() for col in columns}


def derive_monotonicities(frame, columns, target_col, threshold=0.10):
    """Derives monotonicity signs from Spearman correlations.

    Args:
        frame: Training dataframe.
        columns: Modeling columns to evaluate.
        target_col: Binary target column.
        threshold: Absolute correlation cutoff for non-zero monotonicity.

    Returns:
        pd.DataFrame: Feature-wise monotonicity summary.
    """
    rows = []
    for col in columns:
        corr = spearmanr(frame[col], frame[target_col]).statistic
        corr = 0.0 if np.isnan(corr) else float(corr)
        monotonicity = 1 if corr > threshold else -1 if corr < -threshold else 0
        rows.append({"feature": col, "spearman": corr, "abs_spearman": abs(corr), "monotonicity": monotonicity})
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False).reset_index(drop=True)


def monotonicity_name(value):
    """Maps integer monotonicity signs to TensorFlow Lattice labels.

    Args:
        value: One of -1, 0, or 1.

    Returns:
        str: TensorFlow Lattice monotonicity name.
    """
    return {1: "increasing", -1: "decreasing", 0: "none"}[value]


def compute_keypoints(frame, columns, num_keypoints=12):
    """Builds per-feature quantile keypoints for calibration.

    Args:
        frame: Training dataframe.
        columns: Modeling columns.
        num_keypoints: Number of quantile points to request.

    Returns:
        Dict[str, np.ndarray]: Unique float32 keypoints by feature.
    """
    quantiles = np.linspace(0.0, 1.0, num_keypoints)
    keypoints = {}
    for col in columns:
        pts = np.quantile(frame[col], quantiles)
        keypoints[col] = np.unique(np.asarray(pts, dtype=np.float32))
    return keypoints


def build_tfl_model(columns, monotonicity_map, keypoints_map, learning_rate=0.01, l1=0.0):
    """Builds a calibrated linear TensorFlow Lattice model.

    Args:
        columns: Ordered modeling columns.
        monotonicity_map: Feature-to-monotonicity mapping.
        keypoints_map: Feature-to-keypoints mapping.
        learning_rate: Adam learning rate.
        l1: L1 regularization strength on the output layer.

    Returns:
        tuple[keras.Model, dict]: Compiled model and calibrator mapping.
    """
    inputs = {}
    calibrated_outputs = []
    calibrators = {}

    for col in columns:
        inputs[col] = keras.Input(shape=(1,), name=col)
        calibrator = tfl.layers.PWLCalibration(
            input_keypoints=keypoints_map[col],
            dtype=tf.float32,
            monotonicity=monotonicity_name(monotonicity_map[col]),
            output_min=0.0,
            output_max=1.0,
        )
        calibrated_outputs.append(calibrator(inputs[col]))
        calibrators[col] = calibrator

    calibrated_stack = keras.layers.Concatenate(name="calibrated_features")(calibrated_outputs)
    score = keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_constraint=keras.constraints.NonNeg(),
        kernel_regularizer=keras.regularizers.L1(l1),
        name="score",
    )(calibrated_stack)

    model = keras.Model(inputs=inputs, outputs=score)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model, calibrators


def best_threshold(y_true, y_score, resolution=0.01):
    """Selects a cutoff by maximizing F1 over a score grid.

    Args:
        y_true: Binary labels.
        y_score: Predicted probabilities.
        resolution: Step size over the cutoff grid.

    Returns:
        dict: Best cutoff and corresponding F1 score.
    """
    best = {"cutoff": 0.5, "f1": -1.0}
    for cutoff in np.arange(0.01, 1.0, resolution):
        pred = (y_score >= cutoff).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best["f1"]:
            best = {"cutoff": float(cutoff), "f1": float(f1)}
    return best


def confusion_counts(y_true, y_score, cutoff):
    """Computes binary confusion counts at a fixed cutoff.

    Args:
        y_true: Binary labels.
        y_score: Predicted probabilities.
        cutoff: Decision threshold.

    Returns:
        dict: TP, FP, TN, and FN counts.
    """
    pred = (y_score >= cutoff).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def group_fairness_table(frame, group_col, target_col, score_col, cutoff, reference_group, favorable_outcome=0):
    """Summarizes basic fairness metrics by group.

    Args:
        frame: Evaluation dataframe.
        group_col: Group membership column.
        target_col: Binary target column.
        score_col: Predicted probability column.
        cutoff: Decision threshold.
        reference_group: Reference group used for AIR.
        favorable_outcome: Preferred decision label after thresholding.

    Returns:
        pd.DataFrame: Group-level counts and fairness metrics.
    """
    rows = []
    groups = sorted(frame[group_col].dropna().unique())
    ref_frame = frame.loc[frame[group_col] == reference_group].copy()
    ref_decision = (ref_frame[score_col].to_numpy() >= cutoff).astype(int)
    ref_favorable_rate = np.mean(ref_decision == favorable_outcome)

    for group in groups:
        group_frame = frame.loc[frame[group_col] == group].copy()
        y_true = group_frame[target_col].to_numpy()
        y_score = group_frame[score_col].to_numpy()
        counts = confusion_counts(y_true, y_score, cutoff)
        decisions = (y_score >= cutoff).astype(int)
        favorable_rate = np.mean(decisions == favorable_outcome)
        fpr = counts["fp"] / max(counts["fp"] + counts["tn"], 1)
        fnr = counts["fn"] / max(counts["fn"] + counts["tp"], 1)
        rows.append(
            {
                "group": group,
                "n": len(group_frame),
                "favorable_rate": favorable_rate,
                "AIR": favorable_rate / max(ref_favorable_rate, 1e-8),
                "FPR": fpr,
                "FNR": fnr,
            }
        )
    return pd.DataFrame(rows).set_index("group")


def compute_reweighing_weights(frame, group_col, target_col):
    """Builds Kamiran-Calders style reweighing weights.

    Args:
        frame: Training dataframe.
        group_col: Group column used for reweighing.
        target_col: Binary target column.

    Returns:
        np.ndarray: Sample weights aligned to the input rows.
    """
    total = len(frame)
    p_group = frame[group_col].value_counts(normalize=True)
    p_target = frame[target_col].value_counts(normalize=True)
    joint = frame.groupby([group_col, target_col]).size() / total

    weights = []
    for _, row in frame[[group_col, target_col]].iterrows():
        expected = p_group.loc[row[group_col]] * p_target.loc[row[target_col]]
        observed = joint.loc[(row[group_col], row[target_col])]
        weights.append(expected / observed)
    return np.asarray(weights, dtype=np.float32)


def plot_calibrators(calibrators, frame, columns, ncols=3):
    """Plots learned calibrator curves for selected features.

    Args:
        calibrators: Feature-to-calibrator mapping.
        frame: Reference dataframe for plot bounds.
        columns: Features to plot.
        ncols: Number of subplot columns.

    Returns:
        matplotlib.figure.Figure: Figure handle.
    """
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, columns):
        grid = np.linspace(frame[col].quantile(0.01), frame[col].quantile(0.99), 100).astype(np.float32)
        values = calibrators[col](tf.constant(grid.reshape(-1, 1))).numpy().ravel()
        ax.plot(grid, values, linewidth=2)
        ax.set_title(f"calibrator {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("calibrated value")
    for ax in axes[len(columns):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_history(history):
    """Plots training and validation loss.

    Args:
        history: Keras training history.

    Returns:
        matplotlib.figure.Figure: Figure handle.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history["loss"], label="train loss")
    ax.plot(history.history["val_loss"], label="valid loss")
    ax.set_title("training history")
    ax.set_xlabel("epoch")
    ax.legend()
    fig.tight_layout()
    return fig


def score_distribution_plot(scores, y_true, title):
    """Plots predicted score histograms by true class.

    Args:
        scores: Predicted probabilities.
        y_true: Binary labels.
        title: Plot title.

    Returns:
        matplotlib.figure.Figure: Figure handle.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[y_true == 0], bins=30, alpha=0.6, label="actual 0")
    ax.hist(scores[y_true == 1], bins=30, alpha=0.6, label="actual 1")
    ax.set_title(title)
    ax.set_xlabel("predicted delinquency probability")
    ax.legend()
    fig.tight_layout()
    return fig


def roc_plot(y_true, y_score, title):
    """Plots an ROC curve with AUC in the legend.

    Args:
        y_true: Binary labels.
        y_score: Predicted probabilities.
        title: Plot title.

    Returns:
        matplotlib.figure.Figure: Figure handle.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_group_metric_comparison(fairness_tables, metric, title, groups=("black", "hispanic")):
    """Plots one fairness metric across model variants.

    Args:
        fairness_tables: Mapping from label to fairness dataframe.
        metric: Fairness metric column to plot.
        title: Plot title.
        groups: Ordered groups to display.

    Returns:
        matplotlib.figure.Figure: Figure handle.
    """
    labels = list(fairness_tables.keys())
    width = 0.35
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    for index, group in enumerate(groups):
        values = [
            fairness_tables[label].loc[group, metric] if group in fairness_tables[label].index else np.nan
            for label in labels
        ]
        offset = (index - (len(groups) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=group)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric.lower())
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 0), loc=3)
    fig.tight_layout()
    return fig


def train_variant(
    name,
    train_frame,
    valid_frame,
    columns,
    target_col,
    threshold=0.10,
    num_keypoints=12,
    learning_rate=0.01,
    l1=0.0,
    sample_weight=None,
    epochs=150,
    batch_size=512,
):
    """Trains one monotonic TFL variant and scores the holdout sets.

    Args:
        name: Display name for the variant.
        train_frame: Training dataframe.
        valid_frame: Validation dataframe.
        columns: Ordered modeling columns.
        target_col: Binary target column.
        threshold: Correlation cutoff for monotonicity.
        num_keypoints: Calibration keypoint count.
        learning_rate: Adam learning rate.
        l1: Output-layer L1 regularization.
        sample_weight: Optional training weights.
        epochs: Maximum training epochs.
        batch_size: Training batch size.

    Returns:
        TrainedModel: Model plus evaluation artifacts.
    """
    monotonicity_frame = derive_monotonicities(train_frame, columns, target_col, threshold=threshold)
    monotonicity_map = dict(zip(monotonicity_frame["feature"], monotonicity_frame["monotonicity"]))
    keypoints_map = compute_keypoints(train_frame, columns, num_keypoints=num_keypoints)
    model, calibrators = build_tfl_model(columns, monotonicity_map, keypoints_map, learning_rate=learning_rate, l1=l1)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True)
    ]

    history = model.fit(
        frame_to_inputs(train_frame, columns),
        train_frame[target_col].to_numpy(dtype=np.float32),
        validation_data=(frame_to_inputs(valid_frame, columns), valid_frame[target_col].to_numpy(dtype=np.float32)),
        sample_weight=sample_weight,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    valid_scores = model.predict(frame_to_inputs(valid_frame, columns), verbose=0).ravel()
    threshold_result = best_threshold(valid_frame[target_col].to_numpy(), valid_scores)
    valid_auc = roc_auc_score(valid_frame[target_col], valid_scores)

    return TrainedModel(
        name=name,
        model=model,
        calibrators=calibrators,
        feature_cols=list(columns),
        monotonicity_map=monotonicity_map,
        keypoints_map=keypoints_map,
        history=history,
        valid_scores=valid_scores,
        cutoff=threshold_result["cutoff"],
        valid_auc=valid_auc,
    )


def summarize_model(trained_model, valid_frame, target_col, group_col="RACE", reference_group="white"):
    """Builds a compact performance and fairness summary for a model.

    Args:
        trained_model: Trained model artifacts.
        valid_frame: Validation dataframe.
        target_col: Binary target column.
        group_col: Group column for fairness metrics.
        reference_group: Reference group used for AIR.

    Returns:
        tuple[dict, pd.DataFrame]: Summary dict and validation fairness table.
    """
    valid_eval = valid_frame.copy()
    valid_eval["score"] = trained_model.valid_scores

    valid_fair = group_fairness_table(valid_eval, group_col, target_col, "score", trained_model.cutoff, reference_group)

    selected_groups = [g for g in ["black", "hispanic"] if g in valid_fair.index]
    summary = {
        "model": trained_model.name,
        "valid_auc": trained_model.valid_auc,
        "cutoff": trained_model.cutoff,
    }
    for group in selected_groups:
        summary[f"valid_{group}_AIR"] = valid_fair.loc[group, "AIR"]
        summary[f"valid_{group}_FNR"] = valid_fair.loc[group, "FNR"]
    return summary, valid_fair


def select_highest_hispanic_air(candidates):
    """Selects the candidate with the highest hispanic AIR.

    Args:
        candidates: Candidate dictionaries with a `summary` entry.

    Returns:
        dict: Selected candidate entry.
    """
    return max(
        candidates,
        key=lambda entry: (
            entry["summary"].get("valid_hispanic_AIR", float("-inf")),
            entry["summary"]["valid_auc"],
        ),
    )
