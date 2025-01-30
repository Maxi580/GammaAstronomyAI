import pandas as pd
import os
from pathlib import Path


def split_parquet_file(input_file_path: str, output_dir: str = None) -> tuple[str, str]:
    """
    Split a Parquet file into two equal parts.

    Args:
        input_file_path (str): Path to the input Parquet file
        output_dir (str, optional): Directory to save the output files.
                                  If None, uses the same directory as input file.

    Returns:
        tuple[str, str]: Paths to the two output files

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If input file is empty
    """
    # Validate input file
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_file_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the Parquet file
    df = pd.read_parquet(input_file_path)

    # Check if file is empty
    if len(df) == 0:
        raise ValueError("Input Parquet file is empty")

    # Calculate split point
    split_index = len(df) // 4

    # Generate output file names
    input_filename = Path(input_file_path).stem
    file1_path = os.path.join(output_dir, f"{input_filename}_small_part1.parquet")
    file2_path = os.path.join(output_dir, f"{input_filename}_small_part2.parquet")

    # Split and save the files
    df.iloc[:split_index].to_parquet(file1_path, index=False)
    df.iloc[split_index:].to_parquet(file2_path, index=False)

    return file1_path, file2_path


if __name__ == "__main__":
    #import argparse

    #parser = argparse.ArgumentParser(description="Split a Parquet file into two equal parts")
    #parser.add_argument("input_file", help="Path to the input Parquet file")
    #parser.add_argument("--output-dir", help="Directory to save the output files", default=None)

    #args = parser.parse_args()
    input_file = "./magic-protons.parquet"
    output_dir = "."

    try:
        file1, file2 = split_parquet_file(input_file, output_dir)
        print(f"Successfully split the file into:")
        print(f"Part 1: {file1}")
        print(f"Part 2: {file2}")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)