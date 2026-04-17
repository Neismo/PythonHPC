import sys

import pyarrow.parquet as pq
import pyarrow.csv as csv

if __name__ == "__main__":
    data_loc = sys.argv[1]
    table = csv.read_csv(data_loc)
    # save the table as a parquet file
    pq.write_table(table, "data.parquet")