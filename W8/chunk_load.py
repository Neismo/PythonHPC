import sys
import pandas as pd

# SRC=/dtu/projects/02613_2025/data/dmi/2023_01.csv.zip
# chunk_size=1000

if __name__ == "__main__":
    pandas_src = sys.argv[1]
    chunk_size = int(sys.argv[2])

    df = pd.read_csv(pandas_src, chunksize=chunk_size)
    precipitation_sum = 0
    for chunk in df:
        precipitation_sum += chunk.loc[chunk["parameterId"] == "precip_past10min", "value"].sum()

    print(precipitation_sum)