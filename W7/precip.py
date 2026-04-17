import pandas as pd
import sys
from time import perf_counter as time

def total_precip_raw(df):
    total = 0.0
    for i in range(len(df)):
        row = df.iloc[i]
        if row['parameterId'] == 'precip_past10min':
            total += row['value']
    return total


def total_precip_apply(df):
    # with apply method
    return df.apply(lambda row: row['value'] if row['parameterId'] == 'precip_past10min' else 0, axis=1).sum()

def total_precip(df):
    # vectorized approach
    return df.loc[df["parameterId"] == "precip_past10min", "value"].sum()

if __name__ == "__main__":
    DATA_LOC = sys.argv[1] # "/dtu/projects/02613_2025/data/dmi/2023_01.csv.zip"
    df = pd.read_csv(DATA_LOC, compression='zip')

    df = df.sample(frac=0.1)  # Sample 10% of the data

    start_time = time()

    precip = total_precip(df)

    end_time = time()
    print(f"{precip:.4f} in time ({end_time - start_time})")