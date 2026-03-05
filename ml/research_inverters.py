import pandas as pd
import re

file_path = 'Copy of 54-10-EC-8C-14-69.raws.csv'
df = pd.read_csv(file_path, nrows=1000)

print(f"Unique MAC addresses in first 1000 rows: {df['mac'].unique()}")

inverter_cols = [col for col in df.columns if re.search(r'inverters\[\d+\]', col)]
if inverter_cols:
    indices = [int(re.search(r'\[(\d+)\]', col).group(1)) for col in inverter_cols]
    print(f"Inverter indices found: {set(indices)}")
else:
    print("No 'inverters[n]' pattern found in columns.")

meter_cols = [col for col in df.columns if re.search(r'meters\[\d+\]', col)]
if meter_cols:
    indices = [int(re.search(r'\[(\d+)\]', col).group(1)) for col in meter_cols]
    print(f"Meter indices found: {set(indices)}")
