from __future__ import annotations

"""
Error summary statistics utility.

Public API
----------
    error_stats(samples)               -> dict with mean/max/min/std/n
    print_error_summary(channels, ...) -> prints a formatted table to stdout
"""

import math


def error_stats(samples) -> dict:
    """
    Compute summary statistics for a sequence of scalar samples.

    Returns
    -------
    dict with keys: n, mean, max, min, std  (all float; nan when n == 0)
    """
    data = [float(v) for v in samples]
    n = len(data)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "mean": nan, "max": nan, "min": nan, "std": nan}
    mean = sum(data) / n
    var  = sum((v - mean) ** 2 for v in data) / n
    return {
        "n":    n,
        "mean": mean,
        "max":  max(data),
        "min":  min(data),
        "std":  math.sqrt(var),
    }


def print_error_summary(
    channels: dict,
    title: str = "Error Summary",
    unit: str = "",
    indent: int = 2,
):
    """
    Print a formatted statistics table to stdout.

    Parameters
    ----------
    channels : {label: iterable_of_scalars}
        Ordered mapping of channel name → sample values.
    title : str
        Section heading printed above the table.
    unit : str
        Unit string appended to the column header, e.g. "(cm)" or "(deg)".
    indent : int
        Number of leading spaces for each data row.

    Example output
    --------------
      Cartesian Position Error (cm)
      Channel              Mean      Std       Min       Max          N
      X error             -0.34     1.23     -3.46      2.79      1200
      Y error              0.12     0.57     -1.23      1.89      1200
    """
    pad = " " * indent
    col_hdr = f"{'Channel':<24}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}  {'N':>7}"
    unit_str = f" {unit}" if unit else ""

    print(f"\n{pad}{title}{unit_str}")
    print(f"{pad}{col_hdr}")
    print(f"{pad}{'-' * len(col_hdr)}")

    for label, samples in channels.items():
        s = error_stats(samples)
        if s["n"] == 0:
            print(f"{pad}  {label:<22}  {'—':>8}  {'—':>8}  {'—':>8}  {'—':>8}  {'—':>7}")
        else:
            print(
                f"{pad}  {label:<22}"
                f"  {s['mean']:>8.3f}"
                f"  {s['std']:>8.3f}"
                f"  {s['min']:>8.3f}"
                f"  {s['max']:>8.3f}"
                f"  {s['n']:>7d}"
            )
