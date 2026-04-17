import pandas as pd
import pyarrow.parquet as pq
import pyarrow.csv as csv

from time import perf_counter as time

DATA_LOC = "/dtu/projects/02613_2025/data/dmi/2023_01.csv.zip"

def pyarrow_load(path: str):
    pyarrow_table = csv.read_csv(path)
    return pyarrow_table


# This part only relevant for next exercise part
def pyarrow_load_new(path: str):
    pyarrow_table = csv.read_csv(path)
    return pyarrow_table.to_pandas()

print(pyarrow_load(DATA_LOC))