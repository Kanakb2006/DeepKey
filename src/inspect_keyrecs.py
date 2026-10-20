import pandas as pd
from pathlib import Path


FREE_TEXT_PATH = Path("data/raw/free-text.csv")
FIXED_TEXT_PATH = Path("data/raw/fixed-text.csv")


def inspect_dataset(file_path, name):

    print("\n" + "=" * 70)
    print(f"{name} DATASET")
    print("=" * 70)

    # Check file
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return None

    # Load data
    df = pd.read_csv(file_path)

    # ---------------------------------------------------------
    # BASIC INFORMATION
    # ---------------------------------------------------------

    print("\n[1] Shape")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n[2] Column Names")
    for i, column in enumerate(df.columns):
        print(f"{i}: {column}")

    print("\n[3] First 5 Rows")
    print(df.head())

    # ---------------------------------------------------------
    # DATA TYPES
    # ---------------------------------------------------------

    print("\n[4] Data Types")
    print(df.dtypes)

    # ---------------------------------------------------------
    # MISSING VALUES
    # ---------------------------------------------------------

    print("\n[5] Missing Values")

    missing = df.isnull().sum()

    missing_table = pd.DataFrame({
        "Missing_Count": missing,
        "Missing_Percentage": (missing / len(df)) * 100
    })

    print(missing_table[missing_table["Missing_Count"] > 0])

    # ---------------------------------------------------------
    # DUPLICATES
    # ---------------------------------------------------------

    print("\n[6] Duplicate Rows")

    duplicate_count = df.duplicated().sum()

    print("Duplicate rows:", duplicate_count)

    # ---------------------------------------------------------
    # UNIQUE VALUES
    # ---------------------------------------------------------

    print("\n[7] Number of Unique Values")

    unique_values = df.nunique()

    print(unique_values)

    # ---------------------------------------------------------
    # NUMERICAL SUMMARY
    # ---------------------------------------------------------

    print("\n[8] Numerical Summary")

    print(df.describe().T)

    # ---------------------------------------------------------
    # MEMORY
    # ---------------------------------------------------------

    print("\n[9] Memory Usage")

    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    print(f"{memory_mb:.2f} MB")

    return df


# =============================================================
# MAIN
# =============================================================

free_df = inspect_dataset(
    FREE_TEXT_PATH,
    "FREE-TEXT"
)

fixed_df = inspect_dataset(
    FIXED_TEXT_PATH,
    "FIXED-TEXT"
)