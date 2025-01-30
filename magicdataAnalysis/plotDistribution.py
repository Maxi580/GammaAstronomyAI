import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt

def read_parquet_limit(filename, max_rows):
    parquet_file_stream = pq.ParquetFile(filename).iter_batches(batch_size=max_rows)
    
    batch = next(parquet_file_stream)
    
    return batch.to_pandas()

gamma_file = "../magic-gammas-noise.parquet"
proton_file = "../magic-protons-noise.parquet"

gammas = read_parquet_limit(gamma_file, 10000)
protons = read_parquet_limit(proton_file, 10000)

# Extract proton pixels
exploded_m1_protons = protons.noise_image_m1.apply(lambda lst: lst[:1039]).explode()
exploded_m1_protons = exploded_m1_protons.dropna().astype(float)
exploded_m1_protons = exploded_m1_protons[exploded_m1_protons > 0]

exploded_m2_protons = protons.noise_image_m2.apply(lambda lst: lst[:1039]).explode()
exploded_m2_protons = exploded_m2_protons.dropna().astype(float)
exploded_m2_protons = exploded_m2_protons[exploded_m2_protons > 0]

# Extract gamma pixels
exploded_m1_gammas = gammas.noise_image_m1.apply(lambda lst: lst[:1039]).explode()
exploded_m1_gammas = exploded_m1_gammas.dropna().astype(float)
exploded_m1_gammas = exploded_m1_gammas[exploded_m1_gammas > 0]

exploded_m2_gammas = gammas.noise_image_m2.apply(lambda lst: lst[:1039]).explode()
exploded_m2_gammas = exploded_m2_gammas.dropna().astype(float)
exploded_m2_gammas = exploded_m2_gammas[exploded_m2_gammas > 0]

# plot proton pixel distribution
exploded_m1_protons.hist(bins=10000, histtype="step", label="M1")
exploded_m2_protons.hist(bins=10000, histtype="step", label="M2")
plt.xlabel("Value")
plt.ylabel("Counts")
plt.title("Distribution of all pixels - Protons")
plt.xlim(-1, 15)
plt.legend()
plt.savefig("./plots/protons-pixels.png")
plt.clf()


# plot gamma pixel distribution
exploded_m1_gammas.hist(bins=10000, histtype="step", label="M1")
exploded_m2_gammas.hist(bins=10000, histtype="step", label="M2")
plt.xlabel("Value")
plt.ylabel("Counts")
plt.title("Distribution of all pixels - Gammas")
plt.xlim(-1, 15)
plt.legend()
plt.savefig("./plots/gammas-pixels.png")
plt.clf()

# plot m1 pixels
exploded_m1_protons.hist(bins=10000, histtype="step", label="Protons")
exploded_m1_gammas.hist(bins=10000, histtype="step", label="Gammas")
plt.xlabel("Value")
plt.ylabel("Counts")
plt.title("Distribution of all pixels - Protons and Gammas - M1")
plt.xlim(-1, 15)
plt.legend()
plt.savefig("./plots/m1-pixels.png")
plt.clf()

# plot m2 pixels
exploded_m2_protons.hist(bins=10000, histtype="step", label="Protons")
exploded_m2_gammas.hist(bins=10000, histtype="step", label="Gammas")
plt.xlabel("Value")
plt.ylabel("Counts")
plt.title("Distribution of all pixels - Protons and Gammas - M2")
plt.xlim(-1, 15)
plt.legend()
plt.savefig("./plots/m2-pixels.png")
plt.clf()

# plot all pixels
exploded_m1_protons.hist(bins=10000, histtype="step", label="M1 Protons")
exploded_m1_gammas.hist(bins=10000, histtype="step", label="M1 Gammas")
exploded_m2_protons.hist(bins=10000, histtype="step", label="M2 Protons")
exploded_m2_gammas.hist(bins=10000, histtype="step", label="M2 Gammas")
plt.xlabel("Value")
plt.ylabel("Counts")
plt.title("Distribution of all pixels")
plt.xlim(-1, 15)
plt.legend()
plt.savefig('./plots/all-pixels.png')
plt.clf()