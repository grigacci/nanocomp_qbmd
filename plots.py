import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pandas.plotting import parallel_coordinates

# --- Configuration & Styling ---
# Set a style suitable for academic papers
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# File paths
CSV_FILE = './results/pareto_full_sim232.csv'
OUTPUT_DIR = 'plots'  # <--- Output folder name

# Variable Mapping (Mapping internal names to Paper-Ready labels based on your config.toml)
LABELS = {
    # Inputs
    'RW': 'Right Wells (N)',
    'RQWt': 'Right Well Thk (nm)',
    'RQBt': 'Right Barrier Thk (nm)',
    'MQWt': 'Main Well Thk (nm)',
    'LW': 'Left Wells (N)',
    'LQWt': 'Left Well Thk (nm)',
    'LQBt': 'Left Barrier Thk (nm)',
    # Outputs
    'max_energy_eV': 'Peak Energy (eV)',
    'max_photocurrent': 'Max Photocurrent (A)',
    'quality_factor': 'Quality Factor (Q)',
    'prominence': 'Peak Prominence'
}

INPUT_VARS = ['RW', 'RQWt', 'RQBt', 'MQWt', 'LW', 'LQWt', 'LQBt']
OUTPUT_VARS = ['max_photocurrent', 'quality_factor', 'prominence', 'max_energy_eV']

def load_data(filepath):
    """Loads and preprocesses the Pareto CSV data."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} solutions from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found. Please ensure the CSV is in the same directory.")
        return None

def ensure_output_dir():
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def plot_objective_space_matrix(df):
    """
    Creates a PairGrid to visualize trade-offs between all objectives.
    Useful for seeing the shape of the Pareto Front.
    """
    print("Generating Objective Space Matrix...")
    
    # Subset only outputs for this plot
    df_out = df[OUTPUT_VARS].rename(columns=LABELS)
    
    g = sns.PairGrid(df_out, diag_sharey=False, corner=True)
    g.map_lower(sns.scatterplot, s=15, alpha=0.6, edgecolor=None, color="#2b5c8a")
    g.map_diag(sns.histplot, color="#2b5c8a", kde=True)
    
    # --- FIX START ---
    # Iterate through every subplot axis to increase label padding
    for ax in g.axes.flatten():
        if ax: # corner=True leaves upper triangle axes as None, so check existence
            # Increase padding (distance) between ticks and axis labels
            ax.xaxis.labelpad = 20 
            ax.yaxis.labelpad = 20 
    # --- FIX END ---

    g.fig.suptitle('Pareto Front: Objective Trade-offs', y=1.02)
    
    save_path = os.path.join(OUTPUT_DIR, 'pareto_objective_matrix.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_primary_tradeoff(df):
    """
    Plots Photocurrent vs Quality Factor (classic detector trade-off),
    colored by Prominence.
    """
    print("Generating Primary Trade-off Plot...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Check range to decide on log scale.
    # Handle case where min is 0 to avoid RuntimeWarning/Infinity
    min_ph = df['max_photocurrent'].min()
    max_ph = df['max_photocurrent'].max()
    
    if min_ph <= 0:
        # If we have 0s, look at the smallest non-zero value for the ratio check
        non_zero_min = df.loc[df['max_photocurrent'] > 0, 'max_photocurrent'].min()
        if pd.isna(non_zero_min): # If all are 0
            ratio = 1
        else:
            ratio = max_ph / non_zero_min
    else:
        ratio = max_ph / min_ph
        
    use_log_x = ratio > 100
    
    scatter = ax.scatter(
        df['max_photocurrent'], 
        df['quality_factor'], 
        c=df['prominence'], 
        cmap='viridis', 
        s=50, 
        alpha=0.8,
        edgecolor='w',
        linewidth=0.5
    )
    
    ax.set_xlabel(LABELS['max_photocurrent'])
    ax.set_ylabel(LABELS['quality_factor'])
    
    if use_log_x:
        ax.set_xscale('log')
        ax.set_xlabel(LABELS['max_photocurrent'] + ' (Log Scale)')

    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(LABELS['prominence'])
    
    plt.title('Optimization Trade-off: Sensitivity vs Selectivity')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    save_path = os.path.join(OUTPUT_DIR, 'tradeoff_photocurrent_qfactor.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_input_output_correlation(df):
    """
    Heatmap showing correlation between Design Variables (Inputs) 
    and Performance Metrics (Outputs).
    """
    print("Generating Correlation Heatmap...")
    
    # Combine inputs and outputs
    cols = INPUT_VARS + OUTPUT_VARS
    corr_matrix = df[cols].corr(method='spearman') # Spearman captures non-linear monotonic relationships
    
    # Isolate the rectangle: Inputs (rows) vs Outputs (cols)
    heatmap_data = corr_matrix.loc[INPUT_VARS, OUTPUT_VARS]
    
    # Rename index and columns for plotting
    heatmap_data.index = [LABELS[i] for i in heatmap_data.index]
    heatmap_data.columns = [LABELS[i] for i in heatmap_data.columns]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        center=0,
        cbar_kws={'label': 'Spearman Correlation'}
    )
    
    plt.title('Influence of Design Parameters on Detector Performance')
    plt.yticks(rotation=0)
    
    save_path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_parallel_coordinates_normalized(df):
    """
    Normalizes data and plots parallel coordinates to show flow of variables.
    Best for visualizing the 'structure' of high-performing solutions.
    """
    print("Generating Parallel Coordinates Plot...")
    
    # Filter: Let's only plot the top 20% of solutions based on Prominence 
    # to avoid clutter and see what "Good" solutions look like.
    top_percentile = df['prominence'].quantile(0.80)
    df_top = df[df['prominence'] >= top_percentile].copy()
    
    cols_to_plot = INPUT_VARS + ['max_photocurrent', 'quality_factor']
    
    # Normalize data to 0-1 range for visualization
    df_norm = df_top[cols_to_plot].copy()
    for col in df_norm.columns:
        denominator = (df_norm[col].max() - df_norm[col].min())
        if denominator == 0:
            df_norm[col] = 0 # Avoid div by zero if column is constant
        else:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / denominator
    
    # Add a class column for coloring.
    # We try to use nice labels, but fall back to auto-generated intervals 
    # if duplicate values (like many 0s) prevent creating 3 distinct bins.
    try:
        df_norm['Quality'] = pd.qcut(df_top['quality_factor'], q=3, labels=["Low Q", "Med Q", "High Q"])
    except ValueError:
        # Fallback: drop duplicates and let pandas determine the intervals automatically
        # casting to string ensures parallel_coordinates treats it as a discrete class
        print("Warning: Duplicate bin edges detected in Quality Factor (likely many zeros). Falling back to interval labels.")
        df_norm['Quality'] = pd.qcut(df_top['quality_factor'], q=3, duplicates='drop').astype(str)
    
    plt.figure(figsize=(12, 6))
    parallel_coordinates(df_norm, 'Quality', colormap=plt.get_cmap("viridis"), alpha=0.7)
    
    # Fix X-axis labels
    plt.xticks(
        range(len(cols_to_plot)), 
        [LABELS.get(c, c) for c in cols_to_plot], 
        rotation=15
    )
    plt.ylabel('Normalized Value (0-1)')
    plt.title(f'Design Structure of Top 20% Solutions (Sorted by Prominence)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'parallel_coordinates.png')
    plt.savefig(save_path)
    plt.close()

def main():
    ensure_output_dir()
    df = load_data(CSV_FILE)
    if df is not None:
        plot_objective_space_matrix(df)
        plot_primary_tradeoff(df)
        plot_input_output_correlation(df)
        plot_parallel_coordinates_normalized(df)
        print(f"\nAll plots generated successfully in '{OUTPUT_DIR}'!")

if __name__ == "__main__":
    main()