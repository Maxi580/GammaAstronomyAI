import pandas as pd
import os
from pathlib import Path

def combine_parquet_files(
    input_file1: str, input_file2: str, output_file_path: str, columns: list
) -> str:
    """
    Combines two Parquet files, extracting specified columns and appending them in order.

    Args:
        input_file1 (str): Path to the first input Parquet file.
        input_file2 (str): Path to the second input Parquet file.
        output_file_path (str): Path to save the combined Parquet file.
        columns (list): A list of column names to extract from both files.

    Returns:
        str: Path to the combined Parquet file.

    Raises:
        FileNotFoundError: If either input file doesn't exist.
        ValueError: If either input file is empty or if specified columns don't exist.
    """
    # Validate input files
    if not os.path.exists(input_file1):
        raise FileNotFoundError(f"Input file 1 not found: {input_file1}")
    if not os.path.exists(input_file2):
        raise FileNotFoundError(f"Input file 2 not found: {input_file2}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Read the Parquet files
    try:
        df1 = pd.read_parquet(input_file1, columns=columns)
        df2 = pd.read_parquet(input_file2, columns=columns)
    except Exception as e:
        raise ValueError(f"Error reading parquet files: {e}")

    # Check if files are empty
    if df1.empty:
        raise ValueError(f"Input Parquet file 1 is empty: {input_file1}")
    if df2.empty:
        raise ValueError(f"Input Parquet file 2 is empty: {input_file2}")

    # Combine the DataFrames
    df_combined = pd.concat([df1, df2], ignore_index=True)

    # Save the combined DataFrame to a new Parquet file
    df_combined.to_parquet(output_file_path, index=False)

    return output_file_path

if __name__ == "__main__":
    input_file1 = "./magic-gammas_part2.parquet"  # Replace with your first input file
    input_file2 = "./magic-protons_part2.parquet"  # Replace with your second input file
    output_file = "./combined_file.parquet"  # Replace with your desired output file path
    #columns_to_combine = ["run_number", "image_m1", "image_m2"]  # Columns to combine
    columns_to_combine = ["image_m1", "image_m2"]  # Columns to combine

    try:
        combined_file_path = combine_parquet_files(
            input_file1, input_file2, output_file, columns_to_combine
        )
        print(f"Successfully combined files into: {combined_file_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)