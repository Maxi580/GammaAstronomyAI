import pyarrow.parquet as pq
import pandas as pd


def print_parquet_structure(file_path):
    # Load the Parquet file
    table = pq.read_table(file_path)

    # Convert to a Pandas DataFrame for easier inspection
    df = table.to_pandas()

    # Print the structure
    print("Parquet File Structure:")
    print("======================")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns and Data Types:")
    print("======================")
    for column in df.columns:
        print(f"{column}: {df[column].dtype}")


# Example usage
file_path = "./magic-protons_part1.parquet"  # Replace with your Parquet file path
print_parquet_structure(file_path)