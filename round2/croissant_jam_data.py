import pandas as pd
import matplotlib.pyplot as plt

# List of file paths for the three CSV files.
file_paths = [
    '/Users/lianli/Desktop/Prosperity-Lumon/round2/data/prices_round_2_day_-1.csv',
    '/Users/lianli/Desktop/Prosperity-Lumon/round2/data/prices_round_2_day_0.csv',
    '/Users/lianli/Desktop/Prosperity-Lumon/round2/data/prices_round_2_day_1.csv'
]

# Initialize an empty list to store data from each file.
dfs = []

# Loop through each file and load the data.
for file in file_paths:

    df = pd.read_csv(file, delimiter=';')
    
    # Ensure mid_price is numeric.
    df['mid_price'] = pd.to_numeric(df['mid_price'], errors='coerce')
    
    # Print some information from this file to confirm loading.

    
    dfs.append(df)

# Combine all dataframes into one.
combined_df = pd.concat(dfs, ignore_index=True)


# Filter the combined DataFrame to include only CROISSANTS and JAMS.
df_pair = combined_df[combined_df['product'].isin(["CROISSANTS", "JAMS"])]

# Create a pivot table so that each (day, timestamp) has mid_price columns for both products.
df_pivot = df_pair.pivot_table(index=['day', 'timestamp'], columns='product', values='mid_price')

# Drop rows with missing mid_price for either product.
df_pivot = df_pivot.dropna(subset=["CROISSANTS", "JAMS"])


# Calculate the ratio (CROISSANTS / JAMS) for each tick.
df_pivot['ratio'] = df_pivot["CROISSANTS"] / df_pivot["JAMS"]


# Compute overall mean and standard deviation for the ratio.
mean_ratio = df_pivot['ratio'].mean()
std_ratio = df_pivot['ratio'].std()

# Print out the combined statistics.
print("Computed Ratio Statistics (combined over three files):")
print(f"Mean Ratio (CROISSANTS/JAMS): {mean_ratio:.4f}")
print(f"Standard Deviation: {std_ratio:.4f}")
print("=" * 50)

# Optionally, print the full pivoted DataFrame with ratios for further verification.
print("Final pivoted DataFrame with ratios:")
print(df_pivot)

# Compute the z-score for each tick.
df_pivot['z'] = (df_pivot['ratio'] - mean_ratio) / std_ratio

# Print quantiles of the z-score distribution.
quantiles = df_pivot['z'].quantile([0, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 1])
print("Z-score Quantiles:")
print(quantiles)
print("=" * 50)

# Plot a histogram to visualize the distribution of z-scores.
plt.figure(figsize=(8, 4))
plt.hist(df_pivot['z'], bins=50, edgecolor='k', alpha=0.7)
plt.xlabel('Z-Score')
plt.ylabel('Frequency')
plt.title('Distribution of Z-Scores for CROISSANTS/JAMS Ratio')
plt.show()