import os

fams = ["5s", "tmRNA", "tRNA", "srp", "grp1", "RNaseP", "23s", "telomerase", "16s"]
DATA_PATH = "data/fam-fold/"
DATA200_PATH = "data/rnadist/dist_200/"
DATA400_PATH = "data/rnadist/dist_400/"

MODELS_PATH = "models/fam-sim/20_epochs/"
MODELS200_PATH = "models/rnadist/dist_200/20_epochs/"
MODELS400_PATH = "models/rnadist/dist_400/20_epochs/"

# # original training
for fam in fams:
    test_file = f"{DATA_PATH}test_{fam}.csv"
    out_path = f"{MODELS_PATH}{fam}/"
    weigths_path = f"{MODELS_PATH}{fam}/weights.pmt"
    # train = f"sincFold -d cuda train {train_file} --valid-file {valid_file} -o {out_path} -n 5"
    test = (
        f"sincFold -d cuda test {test_file} -w {weigths_path} -o {out_path}test_5e.csv"
    )
    print(f"Running: {test}")
    os.system(test)

for fam in fams:
    test_file = f"{DATA_PATH}test_{fam}.csv"
    out_path = f"{MODELS200_PATH}{fam}/"
    weigths_path = f"{MODELS200_PATH}{fam}/weights.pmt"
    # train = f"sincFold -d cuda train {train_file} --valid-file {valid_file} -o {out_path} -n 5"
    test = (
        f"sincFold -d cuda test {test_file} -w {weigths_path} -o {out_path}test_5e.csv"
    )
    print(f"Running: {test}")
    os.system(test)

for fam in fams:
    test_file = f"{DATA_PATH}test_{fam}.csv"
    out_path = f"{MODELS400_PATH}{fam}/"
    weigths_path = f"{MODELS400_PATH}{fam}/weights.pmt"
    # train = f"sincFold -d cuda train {train_file} --valid-file {valid_file} -o {out_path} -n 5"
    test = (
        f"sincFold -d cuda test {test_file} -w {weigths_path} -o {out_path}test_5e.csv"
    )
    print(f"Running: {test}")
    os.system(test)
