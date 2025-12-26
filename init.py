# Part 1 – Data Exploration
# Libraries needed: pandas, numpy, matplotlib, seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options for better output
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# 1. Load the Heart Disease dataset 
# =============================================================================
df = pd.read_csv('heart.csv')
print("=" * 60)
print("1. Dataset loaded successfully!")
print("=" * 60)

# =============================================================================
# 2. Display dataset information 
# =============================================================================
print("\n" + "=" * 60)
print("2. DATASET INFORMATION")
print("=" * 60)

# First rows of the dataset
print("\n--- First 5 rows of the dataset ---")
print(df.head())

# Dataset shape
print(f"\n--- Dataset Shape ---")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Column names and data types
print("\n--- Column Names and Data Types ---")
print(df.dtypes)

# =============================================================================
# 3. Remove the target column 
# =============================================================================
print("\n" + "=" * 60)
print("3. REMOVING TARGET COLUMN")
print("=" * 60)


print(f"Target column to be deleted : 'target'")
df_features = df.drop(columns=['target'])
print(f"Target column removed. New shape: {df_features.shape}")
print(f"Remaining columns: {df_features.columns.tolist()}")

# =============================================================================
# 4. Compute basic statistics 
# =============================================================================
print("\n" + "=" * 60)
print("4. BASIC STATISTICS")
print("=" * 60)

# Using describe() for comprehensive statistics
 # mean is the average value
 # std is the standard deviation
 # min is the minimum value
 # max is the maximum value
 # count is the number of non-missing values
print("\n--- Basic Statistics (mean, std, min, max) ---")
stats = df_features.describe().loc[['mean', 'std', 'min', 'max','count']]
print(stats)


# =============================================================================
# 5. Analyze feature distributions with visualizations 
# =============================================================================
print("\n" + "=" * 60)
print("5. FEATURE DISTRIBUTION VISUALIZATIONS")
print("=" * 60)

# Use interactive backend for better visualization with zoom/scroll
plt.ion()  # Turn on interactive mode

# --- Create all 3 figures at once (they will appear in separate windows) ---

# --- Figure 1: Histograms ---
print("\nGenerating Histograms (Window 1)...")
fig1, axes1 = plt.subplots(nrows=4, ncols=4, figsize=(14, 10), num='Histograms - Window 1')
axes1 = axes1.flatten()

for i, col in enumerate(df_features.columns):
    if i < len(axes1):
        axes1[i].hist(df_features[col], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes1[i].set_title(f'{col}', fontsize=10, fontweight='bold')
        axes1[i].set_xlabel(col, fontsize=8)
        axes1[i].set_ylabel('Frequency', fontsize=8)
        axes1[i].tick_params(axis='both', labelsize=7)

# Hide empty subplots
for j in range(i + 1, len(axes1)):
    axes1[j].set_visible(False)

fig1.suptitle('Histograms of Features (Use toolbar to zoom/pan)', fontsize=14, fontweight='bold')
fig1.tight_layout()
fig1.savefig('histograms.png', dpi=150)

# --- Figure 2: Box Plots ---
print("Generating Box Plots (Window 2)...")
fig2, axes2 = plt.subplots(nrows=4, ncols=4, figsize=(14, 10), num='Box Plots - Window 2')
axes2 = axes2.flatten()

colors = sns.color_palette("husl", len(df_features.columns))
for i, col in enumerate(df_features.columns):
    if i < len(axes2):
        sns.boxplot(y=df_features[col], ax=axes2[i], color=colors[i])
        axes2[i].set_title(f'{col}', fontsize=10, fontweight='bold')
        axes2[i].set_ylabel('')
        axes2[i].tick_params(axis='both', labelsize=7)

# Hide empty subplots
for j in range(i + 1, len(axes2)):
    axes2[j].set_visible(False)

fig2.suptitle('Box Plots of Features (Use toolbar to zoom/pan)', fontsize=14, fontweight='bold')
fig2.tight_layout()
fig2.savefig('boxplots.png', dpi=150)

# --- Figure 3: Scatter Plots (for selected features) ---
print("Generating Scatter Plots (Window 3)...")

# Select some important features for scatter plots
selected_features = ['age', 'trestbps', 'chol', 'thalach']
selected_features = [f for f in selected_features if f in df_features.columns]

if len(selected_features) >= 2:
    fig3, axes3 = plt.subplots(nrows=2, ncols=3, figsize=(14, 9), num='Scatter Plots - Window 3')
    axes3 = axes3.flatten()
    
    scatter_colors = sns.color_palette("viridis", 6)
    plot_idx = 0
    for i in range(len(selected_features)):
        for j in range(i + 1, len(selected_features)):
            if plot_idx < len(axes3):
                axes3[plot_idx].scatter(df_features[selected_features[i]], 
                                        df_features[selected_features[j]], 
                                        alpha=0.6, edgecolors='white', linewidth=0.3,
                                        c=[scatter_colors[plot_idx]], s=30)
                axes3[plot_idx].set_xlabel(selected_features[i], fontsize=9)
                axes3[plot_idx].set_ylabel(selected_features[j], fontsize=9)
                axes3[plot_idx].set_title(f'{selected_features[i]} vs {selected_features[j]}', 
                                          fontsize=10, fontweight='bold')
                axes3[plot_idx].tick_params(axis='both', labelsize=8)
                plot_idx += 1
    
    # Hide empty subplots
    for k in range(plot_idx, len(axes3)):
        axes3[k].set_visible(False)
    
    fig3.suptitle('Scatter Plots of Selected Features (Use toolbar to zoom/pan)', fontsize=14, fontweight='bold')
    fig3.tight_layout()
    fig3.savefig('scatterplots.png', dpi=150)

# Show all 3 windows simultaneously
print("\n" + "-" * 60)
print("3 VISUALIZATION WINDOWS ARE NOW OPEN!")
print("-" * 60)
print("Tips for interacting with the plots:")
print("  - Use the toolbar at the bottom of each window")
print("  - Pan tool: Click and drag to move around")
print("  - Zoom tool: Draw a rectangle to zoom in")
print("  - Home button: Reset to original view")
print("  - Save button: Save the current view")
print("-" * 60)

plt.show(block=True)  # Block to keep all windows open

# =============================================================================
# 6. Identify and remove missing values 
# =============================================================================
print("\n" + "=" * 60)
print("6. MISSING VALUES ANALYSIS")
print("=" * 60)

# Check for missing values
print("\n--- Missing Values per Column ---")
missing_values = df_features.isnull().sum()
print(missing_values)

print(f"\n--- Total Missing Values: {missing_values.sum()} ---")

# Remove missing values if found with predefined method (here we use dropna)
if missing_values.sum() > 0:
    print(f"\nShape before removing missing values: {df_features.shape}")
    df_features_clean = df_features.dropna()
    print(f"Shape after removing missing values: {df_features_clean.shape}")
    print(f"Rows removed: {df_features.shape[0] - df_features_clean.shape[0]}")
else:
    print("\nNo missing values found in the dataset!")
    df_features_clean = df_features.copy()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY - Part 1 Complete!")
print("=" * 60)
print(f"Original dataset shape: {df.shape}")
print(f"Features dataset shape (without target): {df_features.shape}")
print(f"Clean dataset shape (after handling missing values): {df_features_clean.shape}")
print("\nVisualization files saved:")
print("- histograms.png")
print("- boxplots.png")
print("- scatterplots.png")
