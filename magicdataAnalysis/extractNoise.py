import pandas as pd
import os
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path



def read_parquet_limit(filename, max_rows):
    parquet_file_stream = pq.ParquetFile(filename).iter_batches(batch_size=max_rows)
    
    batch = next(parquet_file_stream)
    
    return batch.to_pandas()

def extract_noise(input_file_path: str):
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    schema = pa.schema([
        ('noise_image_m1', pa.list_(pa.float64())),
        ('noise_image_m2', pa.list_(pa.float64())),
        ('event_number', pa.uint32()),
    ])
    union_schema = pa.unify_schemas([schema, pq.read_schema(input_file_path)])
    
    input_filename = Path(input_file_path).stem
    output_dir = Path(input_file_path).parent
    parquet_file_stream = pq.ParquetFile(input_file_path).iter_batches(batch_size=10000)
    parquet_write = pq.ParquetWriter(os.path.join(output_dir, f'{input_filename}-noise.parquet'), schema=union_schema)

    for batch in parquet_file_stream:
        df = batch.to_pandas()
        
        df['noise_image_m1'] = [a_i - b_i for a_i, b_i in zip(df['image_m1'], df['clean_image_m1'])]
        df['noise_image_m2'] = [a_i - b_i for a_i, b_i in zip(df['image_m2'], df['clean_image_m2'])]
        
        transformed_batch = pa.RecordBatch.from_pandas(df, schema=union_schema)
        parquet_write.write_batch(transformed_batch)



if __name__ == "__main__":
    input_file = "../magic-gammas.parquet"
    
    extract_noise(input_file)
