"""Reusable EDA plotting helpers for the notebook workflow."""

import math
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_LABEL_MAP = {
    0: "Bearish (0)",
    1: "Bullish (1)",
    2: "Neutral (2)",
}

DEFAULT_LABEL_PALETTE = {
    0: "#3586B5",
    1: "steelblue",
    2: "#3052B5",
}


def plot_numeric_distributions(
    data: pd.DataFrame,
    numeric_cols: Sequence[str],
    title: str,
    *,
    bins: int = 30,
    n_cols: int = 2,
    color: str = "steelblue",
    figsize: tuple[float, float] | None = None,
):
    """Plot overall numeric distributions with mean and median reference lines."""
    cols = list(numeric_cols)
    if not cols:
        raise ValueError("numeric_cols must contain at least one column.")

    n_rows = math.ceil(len(cols) / n_cols)
    fig, axes_grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize or (12, n_rows * 4),
        squeeze=False,
    )
    axes = axes_grid.flatten()

    for i, col in enumerate(cols):
        _plot_distribution(data, col, axes[i], bins=bins, color=color)
        axes[i].set_title(col, fontsize=11, fontweight="bold")

    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)
    sns.despine()

    return fig, axes


def plot_numeric_distributions_by_label(
    data: pd.DataFrame,
    numeric_cols: Sequence[str],
    *,
    label_col: str = "label",
    label_map: Mapping[object, str] | None = None,
    title: str,
    bins: int = 30,
    color: str = "steelblue",
    figsize: tuple[float, float] | None = None,
):
    """Plot numeric distributions split across label categories."""
    cols = list(numeric_cols)
    labels = _sorted_labels(data, label_col)
    labels_by_value = label_map or DEFAULT_LABEL_MAP

    if not cols:
        raise ValueError("numeric_cols must contain at least one column.")

    fig, axes = plt.subplots(
        len(cols),
        len(labels),
        figsize=figsize or (5.35 * len(labels), 4 * len(cols)),
        squeeze=False,
    )

    for row, col in enumerate(cols):
        for column, label in enumerate(labels):
            ax = axes[row, column]
            subset = data.loc[data[label_col] == label]
            _plot_distribution(subset, col, ax, bins=bins, color=color)
            ax.set_title(
                f"{col} | {labels_by_value.get(label, label)}",
                fontsize=11,
                fontweight="bold",
            )

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    sns.despine()
    plt.tight_layout()

    return fig, axes


def plot_relative_frequency_by_label(
    data: pd.DataFrame,
    value_col: str,
    *,
    label_col: str = "label",
    label_map: Mapping[object, str] | None = None,
    title: str,
    panel_title: str | None = None,
    x_label: str | None = None,
    color: str = "steelblue",
    figsize: tuple[float, float] | None = None,
):
    """Plot value relative frequencies separately for each label category."""
    labels = _sorted_labels(data, label_col)
    labels_by_value = label_map or DEFAULT_LABEL_MAP

    fig, axes_grid = plt.subplots(
        1,
        len(labels),
        figsize=figsize or (6 * len(labels), 5),
        squeeze=False,
    )
    axes = axes_grid.flatten()

    for ax, label in zip(axes, labels):
        subset = data.loc[data[label_col] == label]
        rel_freq = subset[value_col].value_counts(normalize=True).sort_index().mul(100)

        sns.barplot(
            x=rel_freq.index,
            y=rel_freq.values,
            color=color,
            edgecolor="white",
            alpha=0.8,
            ax=ax,
        )

        ax.set_title(
            f"{panel_title or value_col} | {labels_by_value.get(label, label)}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel(x_label or value_col, fontsize=10)
        ax.set_ylabel("Relative Frequency (%)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        _add_bar_labels(ax)

    sns.despine()
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig, axes


def plot_boolean_presence_by_label(
    data: pd.DataFrame,
    boolean_cols: Sequence[str],
    *,
    label_col: str = "label",
    label_map: Mapping[object, str] | None = None,
    titles: Sequence[str] | None = None,
    x_labels: Sequence[str] | None = None,
    palette: Mapping[object, str] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Plot boolean-column relative frequencies across label categories."""
    cols = list(boolean_cols)
    if not cols:
        raise ValueError("boolean_cols must contain at least one column.")

    labels = _sorted_labels(data, label_col)
    labels_by_value = label_map or DEFAULT_LABEL_MAP
    colors_by_label = palette or DEFAULT_LABEL_PALETTE
    colors = [colors_by_label.get(label, "steelblue") for label in labels]

    fig, axes_grid = plt.subplots(
        1,
        len(cols),
        figsize=figsize or (7 * len(cols), 5),
        squeeze=False,
    )
    axes = axes_grid.flatten()

    for ax, col, col_title, x_label in zip(
        axes,
        cols,
        titles or cols,
        x_labels or cols,
        strict=True,
    ):
        rel_freq = pd.crosstab(data[col], data[label_col], normalize="columns") * 100
        rel_freq = rel_freq.reindex(columns=labels, fill_value=0)

        rel_freq.plot(
            kind="bar",
            ax=ax,
            color=colors,
            edgecolor="white",
            alpha=0.8,
        )

        ax.set_title(col_title, fontsize=14, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Relative Frequency (%)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(
            title="Sentiment",
            labels=[labels_by_value.get(label, label) for label in labels],
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        _add_bar_labels(ax)

    sns.despine()
    plt.tight_layout()

    return fig, axes


def _plot_distribution(
    data: pd.DataFrame,
    col: str,
    ax,
    *,
    bins: int,
    color: str,
) -> None:
    mean = data[col].mean()
    median = data[col].median()

    sns.histplot(
        data=data,
        x=col,
        ax=ax,
        bins=bins,
        kde=True,
        stat="probability",
        color=color,
        edgecolor="white",
        alpha=0.8,
    )

    ax.axvline(mean, linestyle="--", linewidth=1.8, label=f"Mean = {mean:.2f}")
    ax.axvline(median, linestyle="-", linewidth=1.8, label=f"Median = {median:.2f}")
    ax.set_xlabel("")
    ax.set_ylabel("Relative frequency", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def _sorted_labels(data: pd.DataFrame, label_col: str) -> list[object]:
    labels = sorted(data[label_col].dropna().unique())
    if not labels:
        raise ValueError(f"{label_col!r} must contain at least one non-null label.")

    return labels


def _add_bar_labels(ax) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=8, padding=2)
