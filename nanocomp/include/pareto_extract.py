#!/usr/bin/env python3
"""
pareto_extract.py

Usage:
    python pareto_extract.py path/to/ga_output.csv

Assumptions:
- CSV has columns:
  RW,RQWt,RQBt,MQWt,LW,LQWt,LQBt,
  max_energy_eV,max_photocurrent,quality_factor,prominence,constraint

- You can edit the 'objectives' list and 'direction' mapping
  below to set which metrics are used to compute Pareto optimality
  and whether to 'max' or 'min' each metric.

Output:
- writes a CSV file with suffix "_pareto.csv"
- prints number of pareto points and saves a couple of simple plots
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# -------------------- User-editable settings --------------------
# Path to CSV will be passed as first CLI arg
# Which metric columns are used to compute Pareto front:
objectives = [
    "max_photocurrent",
    "quality_factor",
    "prominence",
    "max_energy_eV",
    "constraint",
]

# For each objective choose 'max' (bigger is better) or 'min' (smaller is better).
# Defaults chosen conservatively based on typical preferences:
# - max_photocurrent: maximize (bigger photocurrent better)
# - quality_factor: maximize
# - prominence: minimize (you mentioned "menor corrente de escuro = melhor")
# - max_energy_eV: minimize by default (if you want to maximize energy, change to 'max')
# - constraint: minimize (treat lower constraint value as better; if this column
#   is an infeasibility indicator, see 'feasibility_threshold' below)
direction = {
    "max_photocurrent": "max",
    "quality_factor": "max",
    "prominence": "min",
    "max_energy_eV": "min",
    "constraint": "min",
}

# If constraint is actually a feasibility metric where values <= threshold are feasible,
# set feasibility_filter_on=True and adjust threshold. If you don't want to filter,
# set feasibility_filter_on=False.
feasibility_filter_on = True
feasibility_column = "constraint"
feasibility_threshold = 0.0  # keep rows where constraint <= threshold

# Plotting options
save_plots = True
plot_dir = "pareto_plots"
# ----------------------------------------------------------------

def read_csv(path):
    df = pd.read_csv(path)
    return df

def build_objective_matrix(df, objectives, direction):
    """
    Convert chosen objectives to a numeric matrix where *higher is better* for all columns.
    For an objective that should be minimized, we multiply by -1 so that larger transformed
    values mean better performance.
    Returns (obj_matrix, used_objective_names)
    """
    arrs = []
    used = []
    for obj in objectives:
        if obj not in df.columns:
            raise ValueError(f"Objective column '{obj}' not found in CSV columns: {list(df.columns)}")
        col = df[obj].astype(float).to_numpy(copy=True)
        dirn = direction.get(obj, "max")
        if dirn not in ("max", "min"):
            raise ValueError(f"direction for {obj} must be 'max' or 'min'")
        if dirn == "max":
            arrs.append(col)
        else:
            arrs.append(-col)  # flip sign so larger is better
        used.append(obj)
    mat = np.vstack(arrs).T  # shape (n_rows, n_objectives)
    return mat, used

def is_pareto_efficient(scores):
    """
    Vectorized check for Pareto efficiency (non-dominated points).
    scores: 2D numpy array where larger is better.
    Returns boolean mask of length n_points where True indicates Pareto-optimal.
    """
    # Implementation that works well for moderate N (few thousands).
    # Complexity is O(N^2 * M) where M is num objectives.
    n_points = scores.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not is_efficient[i]:
            continue
        # Any point j that is dominated by i should be marked False
        # i dominates j if scores[i] >= scores[j] for all dims AND > for at least one dim
        # We will compare i against all j != i
        comp = scores >= scores[i]  # shape (n_points, M) boolean: for each j, dim >= i?
        # For j dominated by i, all True in comp[j] and at least one strict >
        all_ge = np.all(comp, axis=1)
        strictly_greater = np.any(scores > scores[i], axis=1)
        dominated_by_i = all_ge & strictly_greater
        dominated_by_i[i] = False  # a point doesn't dominate itself
        is_efficient[dominated_by_i] = False
    return is_efficient

def main(csv_path):
    if not os.path.exists(csv_path):
        print("File not found:", csv_path)
        sys.exit(1)

    df = read_csv(csv_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    # apply feasibility filter if enabled
    if feasibility_filter_on:
        if feasibility_column not in df.columns:
            raise ValueError(f"Feasibility column '{feasibility_column}' not in CSV.")
        feasible_mask = df[feasibility_column].astype(float) <= feasibility_threshold
        n_before = len(df)
        df = df.loc[feasible_mask].reset_index(drop=True)
        n_after = len(df)
        print(f"Applied feasibility filter: kept {n_after}/{n_before} rows where {feasibility_column} <= {feasibility_threshold}")

    # Build objective matrix (higher-is-better)
    obj_matrix, used_obj_names = build_objective_matrix(df, objectives, direction)
    print("Using objectives (transformed to 'higher is better'):", used_obj_names)
    # compute pareto mask
    pareto_mask = is_pareto_efficient(obj_matrix)
    pareto_df = df.loc[pareto_mask].copy().reset_index(drop=True)

    print(f"Pareto front size: {len(pareto_df)} points out of {len(df)} (after feasibility filter).")

    # Add a column to indicate pareto status in original (non-filtered) index
    # if feasibility filter was on, we can still output pareto_df alone
    # Save pareto CSV
    base, ext = os.path.splitext(csv_path)
    out_path = base + "_pareto.csv"
    pareto_df.to_csv(out_path, index=False)
    print(f"Wrote Pareto front CSV to: {out_path}")

    # Simple diagnostics: save Pareto indices and first rows
    pareto_indices = np.where(pareto_mask)[0]
    print("Pareto indices (relative to filtered dataframe):", pareto_indices[:50].tolist())
    if len(pareto_df) > 0:
        print("First 5 Pareto rows:")
        print(pareto_df.head(5).to_string(index=False))

    # Optional plotting
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        # 1) scatter matrix for the objectives with pareto points highlighted
        plot_cols = used_obj_names
        try:
            sm = scatter_matrix(df[plot_cols], alpha=0.4, figsize=(12, 12), diagonal="kde")
            # highlight pareto points in red
            for ax in sm.ravel():
                # each scatter plot is a PathCollection at ax.collections[0] (pandas plotting detail)
                ax.scatter(pareto_df[plot_cols[0]].values,
                           pareto_df[plot_cols[1] if len(plot_cols) > 1 else plot_cols[0]].values,
                           s=40, facecolors='none', edgecolors='r', label='pareto' if ax is sm.ravel()[0] else "")
            plt.suptitle("Scatter matrix of objectives (pareto in red)")
            scatter_path = os.path.join(plot_dir, "scatter_matrix_objectives.png")
            plt.savefig(scatter_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved scatter matrix to {scatter_path}")
        except Exception as e:
            print("Warning: scatter matrix plot failed:", e)

        # 2) pairwise 2D plots between top two objectives (if at least 2)
        if len(used_obj_names) >= 2:
            xcol, ycol = used_obj_names[0], used_obj_names[1]
            plt.figure(figsize=(6,5))
            plt.scatter(df[xcol], df[ycol], alpha=0.3, label='population')
            plt.scatter(pareto_df[xcol], pareto_df[ycol], edgecolors='r', facecolors='none', s=80, label='pareto')
            plt.xlabel(xcol); plt.ylabel(ycol); plt.legend()
            plt.title(f"{ycol} vs {xcol} (Pareto highlighted)")
            pair_path = os.path.join(plot_dir, f"{xcol}_vs_{ycol}_pareto.png")
            plt.savefig(pair_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved 2D pareto plot to {pair_path}")

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pareto_extract.py path/to/ga_output.csv")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
