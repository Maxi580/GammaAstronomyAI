import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt

def read_parquet_limit(filename, max_rows):
    parquet_file_stream = pq.ParquetFile(filename).iter_batches(batch_size=max_rows)
    
    batch = next(parquet_file_stream)
    
    return batch.to_pandas()

gamma_file = "../magic-gammas-new-1.parquet"
proton_file = "../magic-protons.parquet"

gammas = read_parquet_limit(gamma_file, 10000)
protons = read_parquet_limit(proton_file, 10000)

# Extract energy levels used as simulation input
true_energy_protons = protons.true_energy
true_energy_gammas  = gammas.true_energy


print("MAX Protons", max(true_energy_protons))
print("MAX Gammas", max(true_energy_gammas))
# plot proton pixel distribution
true_energy_protons.hist(bins=100, histtype="step", density=True, label="Protons")
true_energy_gammas.hist(bins=100, histtype="step", density=True, label="Gammas")
plt.xlabel("Value")
plt.ylabel("Counts")
plt.title("Distribution of energy levels used as input for simulations")
# plt.xlim(-1, 15)
plt.legend()
plt.savefig("./plots/energy-levels.png")
plt.clf()