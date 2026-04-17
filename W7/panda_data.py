import pandas as pd

DATA_LOC = "/dtu/projects/02613_2025/data/dmi/2023_01.csv.zip"

dataframe = pd.read_csv(DATA_LOC, compression='zip')

# Check memory size of the dataframe
print(dataframe.memory_usage(deep=True).sum())

def df_memsize(pandas_df: pd.DataFrame) -> int:
    """Calculate the memory size of a pandas dataframe in bytes."""
    return pandas_df.memory_usage(deep=True).sum()

def summarize_columns(df: pd.DataFrame):
    print(pd.DataFrame([(c,df[c].dtype,len(df[c].unique()),df[c].memory_usage(deep=True) // (1024**2)) 
                        for c 
                        in df.columns], columns=['name', 'dtype', 'unique', 'size (MB)']))
    print('Total size:', df.memory_usage(deep=True).sum() / 1024**2, 'MB')