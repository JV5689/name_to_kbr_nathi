import pandas as pd
import numpy as np

file_path = 'Copy of 54-10-EC-8C-14-69.raws.csv'

# Read just the header to get columns
df_head = pd.read_csv(file_path, nrows=5)
columns = list(df_head.columns)

print(f"Total Columns: {len(columns)}")
print("Columns list:")
for col in columns:
    print(f"- {col}")

# Get info on a larger sample to see data types and nulls
df_sample = pd.read_csv(file_path, nrows=1000)
print("\nData Sample Info:")
print(df_sample.info())

# Look for 'status', 'error', 'failure', or 'fault' in column names
target_keywords = ['status', 'error', 'failure', 'fault', 'alarm', 'state']
potential_targets = [col for col in columns if any(kw in col.lower() for kw in target_keywords)]
print("\nPotential Target Columns:")
for target in potential_targets:
    print(f"- {target}: {df_sample[target].unique() if target in df_sample.columns else 'N/A'}")

# Save the column list to a file for easier reference
with open('csv_columns.txt', 'w') as f:
    for col in columns:
        f.write(f"{col}\n")
