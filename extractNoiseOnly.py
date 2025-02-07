import pandas as pd
import os
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import random, math



def read_parquet_limit(filename, max_rows):
    parquet_file_stream = pq.ParquetFile(filename).iter_batches(batch_size=max_rows)
    batch = next(parquet_file_stream)
    return batch.to_pandas()

def chunks(xs, n):
    n = max(1, n)
    return (xs[i:i+n] for i in range(0, len(xs), n))


def extract_noise(input_file_path: str, label: str):
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    schema = pa.schema([
        ('noise_m1', pa.list_(pa.float32())),
        ('noise_m2', pa.list_(pa.float32())),
        ('label', pa.string()),
    ])

    input_filename = Path(input_file_path).stem
    output_dir = Path(input_file_path).parent
    parquet_file_stream = pq.ParquetFile(input_file_path).iter_batches(batch_size=10000)
    parquet_write = pq.ParquetWriter(os.path.join(output_dir, f'{input_filename}-noiseonly.parquet'), schema=schema)

    noise_arr_m1 = []
    noise_arr_m2 = []
    for batch in parquet_file_stream:
        df = batch.to_pandas()

        noise_m1 = [a_i - b_i for a_i, b_i in zip(df['image_m1'], df['clean_image_m1'])]
        noise_m2 = [a_i - b_i for a_i, b_i in zip(df['image_m2'], df['clean_image_m2'])]

        for m1, m2 in zip(noise_m1, noise_m2):
            noise_arr_m1 += [x for x in m1 if x != 0]
            noise_arr_m2 += [x for x in m2 if x != 0]

    print("Total noise M1:", len(noise_arr_m1))
    print("Total noise M2:", len(noise_arr_m2))

    random.shuffle(noise_arr_m1)
    random.shuffle(noise_arr_m2)

    chunk_size = 100

    cut = (math.floor(min(len(noise_arr_m1), len(noise_arr_m2)) / chunk_size)) * chunk_size

    print("Cutting to:", cut)
    noise_arr_m1 = noise_arr_m1[:cut]
    noise_arr_m2 = noise_arr_m2[:cut]

    noise_dicts = [
        {
            "noise_m1": x1,
            "noise_m2": x2,
            "label": label
        } for x1, x2 in zip(chunks(noise_arr_m1, chunk_size), chunks(noise_arr_m2, chunk_size))
    ]

    print("ITEMS:", len(noise_dicts))


    output = pa.Table.from_pylist(noise_dicts, schema=schema)
    parquet_write.write(output)



if __name__ == "__main__":
    input_file = "./magic-gammas.parquet"

    extract_noise(input_file, "gamma")