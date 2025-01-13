import pandas as pd
import os
from pathlib import Path

def extract_columns_to_parquet(
    input_file_path: str, columns: list, output_dir: str = None
) -> str:
    """
    Extracts specified columns from a Parquet file and saves them into a new Parquet file.

    Args:
        input_file_path (str): Path to the input Parquet file.
        columns (list): A list of column names to extract.
        output_dir (str, optional): Directory to save the output file.
                                   If None, uses the same directory as the input file.

    Returns:
        str: Path to the newly created Parquet file.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        ValueError: If the input file is empty or if any specified column doesn't exist.
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
    try:
        df = pd.read_parquet(input_file_path)
    except Exception as e:
        raise ValueError(f"Error reading parquet file: {e}")

    # Check if file is empty
    if df.empty:
        raise ValueError("Input Parquet file is empty")

    # Check if all specified columns exist
    if not all(col in df.columns for col in columns):
        missing_cols = [col for col in columns if col not in df.columns]
        raise ValueError(
            f"Some specified columns are not found in the input file: {missing_cols}"
        )

    # Extract the specified columns
    df_extracted = df[columns]

    # Generate output file name
    input_filename = Path(input_file_path).stem
    output_file_path = os.path.join(
        output_dir, f"{input_filename}_extracted.parquet"
    )

    # Save the extracted columns to a new Parquet file
    df_extracted.to_parquet(output_file_path, index=False)

    return output_file_path

if __name__ == "__main__":
    input_file = "./magic-gammas_part2.parquet"  # Replace with your input file
    output_dir = "."  # Replace with your desired output directory
    columns_to_extract = ["image_m1", "image_m2"]  # Add more columns if needed

    try:
        output_file = extract_columns_to_parquet(
            input_file, columns_to_extract, output_dir
        )
        print(f"Successfully extracted columns to: {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)