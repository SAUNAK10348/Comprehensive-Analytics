"""Simple analytics helpers for keyword exploration."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from typing import Iterable, List


def plot_keyword_frequency(texts: Iterable[str]):
    words = " ".join(texts).lower().split()
    freq = pd.Series(words).value_counts().head(10)
    ax = freq.plot(kind="bar")
    ax.set_title("Top Keywords")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Keyword")
    fig = ax.get_figure()
    fig.tight_layout()
    return fig


__all__ = ["plot_keyword_frequency"]
