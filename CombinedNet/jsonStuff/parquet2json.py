import json
import os
import pandas as pd
from pathlib import Path

PROTON_LABEL: str = "proton"
GAMMA_LABEL: str = "gamma"


def inspect_parquet(file_path):
    df = pd.read_parquet(file_path)

    print("=== Parquet File Structure ===")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col} (type: {df[col].dtype})")

    print("\nFirst row data:")
    print(df.iloc[0].to_dict())

    print("\nDataset Info:")
    print(df.info())

    return df


def convert_parquet_to_json_txt(parquet_file: str, output_dir: str, label: str) -> None:
    arrays_dir = os.path.join(output_dir, "arrays")
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(arrays_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    df = pd.read_parquet(parquet_file)
    prefix = Path(parquet_file).stem

    for idx in range(len(df)):
        row = df.iloc[idx]

        if len(row['image_m1']) != 1183:
            print(f"Warning: Row {idx} has unexpected array length: {len(row['clean_image_m1'])}")
            continue

        base_name = f"{prefix}_{idx}"
        array_file = os.path.join(arrays_dir, f"{base_name}.json")
        label_file = os.path.join(annotations_dir, f"{base_name}.txt")

        data = {
            "images": {
                "raw": {
                    "m1": row['image_m1'].tolist(),
                    "m2": row['image_m2'].tolist()
                },
            },
            "features": {
                "true_parameters": {
                    "energy": float(row['true_energy']),
                    "theta": float(row['true_theta']),
                    "phi": float(row['true_phi']),
                    "telescope_theta": float(row['true_telescope_theta']),
                    "telescope_phi": float(row['true_telescope_phi']),
                    "first_interaction_height": float(row['true_first_interaction_height']),
                    "impact_m1": float(row['true_impact_m1']),
                    "impact_m2": float(row['true_impact_m2'])
                },
                "hillas_m1": {
                    "length": float(row['hillas_length_m1']),
                    "width": float(row['hillas_width_m1']),
                    "delta": float(row['hillas_delta_m1']),
                    "size": float(row['hillas_size_m1']),
                    "cog_x": float(row['hillas_cog_x_m1']),
                    "cog_y": float(row['hillas_cog_y_m1']),
                    "sin_delta": float(row['hillas_sin_delta_m1']),
                    "cos_delta": float(row['hillas_cos_delta_m1'])
                },
                "hillas_m2": {
                    "length": float(row['hillas_length_m2']),
                    "width": float(row['hillas_width_m2']),
                    "delta": float(row['hillas_delta_m2']),
                    "size": float(row['hillas_size_m2']),
                    "cog_x": float(row['hillas_cog_x_m2']),
                    "cog_y": float(row['hillas_cog_y_m2']),
                    "sin_delta": float(row['hillas_sin_delta_m2']),
                    "cos_delta": float(row['hillas_cos_delta_m2'])
                },
                "stereo": {
                    "direction_x": float(row['stereo_direction_x']),
                    "direction_y": float(row['stereo_direction_y']),
                    "zenith": float(row['stereo_zenith']),
                    "azimuth": float(row['stereo_azimuth']),
                    "dec": float(row['stereo_dec']),
                    "ra": float(row['stereo_ra']),
                    "theta2": float(row['stereo_theta2']),
                    "core_x": float(row['stereo_core_x']),
                    "core_y": float(row['stereo_core_y']),
                    "impact_m1": float(row['stereo_impact_m1']),
                    "impact_m2": float(row['stereo_impact_m2']),
                    "impact_azimuth_m1": float(row['stereo_impact_azimuth_m1']),
                    "impact_azimuth_m2": float(row['stereo_impact_azimuth_m2']),
                    "shower_max_height": float(row['stereo_shower_max_height']),
                    "xmax": float(row['stereo_xmax']),
                    "cherenkov_radius": float(row['stereo_cherenkov_radius']),
                    "cherenkov_density": float(row['stereo_cherenkov_density']),
                    "baseline_phi_m1": float(row['stereo_baseline_phi_m1']),
                    "baseline_phi_m2": float(row['stereo_baseline_phi_m2']),
                    "image_angle": float(row['stereo_image_angle']),
                    "cos_between_shower": float(row['stereo_cos_between_shower'])
                },
                "pointing": {
                    "zenith": float(row['pointing_zenith']),
                    "azimuth": float(row['pointing_azimuth'])
                },
                "time_gradient": {
                    "m1": float(row['time_gradient_m1']),
                    "m2": float(row['time_gradient_m2'])
                },
                "source_m1": {
                    "alpha": float(row['source_alpha_m1']),
                    "dist": float(row['source_dist_m1']),
                    "cos_delta_alpha": float(row['source_cos_delta_alpha_m1']),
                    "dca": float(row['source_dca_m1']),
                    "dca_delta": float(row['source_dca_delta_m1'])
                },
                "source_m2": {
                    "alpha": float(row['source_alpha_m2']),
                    "dist": float(row['source_dist_m2']),
                    "cos_delta_alpha": float(row['source_cos_delta_alpha_m2']),
                    "dca": float(row['source_dca_m2']),
                    "dca_delta": float(row['source_dca_delta_m2'])
                }
            }
        }

        with open(array_file, 'w') as f:
            json.dump(data, f)

        with open(label_file, 'w') as f:
            f.write(str(label))

        if idx % 1000 == 0:
            print(f"Processed {idx} rows...")


if __name__ == "__main__":
    # inspect_parquet("magic-protons.parquet")
    convert_parquet_to_json_txt("../../magic-protons.parquet", "../datasets/magic_protons", PROTON_LABEL)
