import pandas as pd

# Path to raw CSV
RAW_DATA_PATH = "./data/raw/Customer-Churn.csv"

def load_dataset(path=RAW_DATA_PATH):
    """
    Load dataset from CSV.
    """
    df = pd.read_csv(path)
    return df

def df_overview(df):
    """
    Print dataset overview: shape, dtypes, missing values.
    """
    print("\n=== Shape ===")
    print(df.shape)

    print("\n=== Dtypes ===")
    print(df.dtypes)

    print("\n=== Missing Values ===")
    print(df.isna().sum().sort_values(ascending=False).head(20))

# xecution
if __name__ == "__main__":
    df = load_dataset()
    df_overview(df)

