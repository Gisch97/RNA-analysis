import os

fams = ["5s", "tmRNA", "tRNA", "srp", "grp1", "RNaseP", "23s", "telomerase", "16s"]
DATA_PATH = "data/fam-fold/"
DATA200_PATH = "data/rnadist/dist_200/"
DATA400_PATH = "data/rnadist/dist_400/"

MODELS_PATH = "models/fam-sim/10_epochs/"
MODELS200_PATH = "models/rnadist/dist_200/10_epochs/"
MODELS400_PATH = "models/rnadist/dist_400/10_epochs/"

SAVE_PATH = "models/fam-sim/20_epochs/"
SAVE200_PATH = "models/rnadist/dist_200/20_epochs/"
SAVE400_PATH = "models/rnadist/dist_400/20_epochs/"

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SAVE200_PATH, exist_ok=True)
os.makedirs(SAVE400_PATH, exist_ok=True)

# # original training
# for fam in fams:
for fam in ["grp1", "RNaseP", "23s", "telomerase", "16s"]:
    train_file = f"{DATA_PATH}train_{fam}.csv"
    valid_file = f"{DATA_PATH}valid_{fam}.csv"
    weigths_path = f"{MODELS_PATH}{fam}/weights.pmt"
    out_path = f"{SAVE_PATH}{fam}/"

    train = f"sincFold -d cuda train {train_file} --valid-file {valid_file} -o {out_path} -n 10 -w {weigths_path}"
    print(f"Running: {train}")
    os.system(train)

# # training on rnadist datasets max 200
for fam in fams:
    train_file = f"{DATA200_PATH}train_{fam}.csv"
    valid_file = f"{DATA200_PATH}valid_{fam}.csv"
    out_path = f"{SAVE200_PATH}{fam}/"
    weigths_path = f"{MODELS200_PATH}{fam}/weights.pmt"

    train = f"sincFold -d cuda train {train_file} --valid-file {valid_file} -o {out_path} -n 10 -w {weigths_path}"
    print(f"Running: {train}")
    os.system(train)


# training on rnadist datasets max 400
for fam in fams:
    train_file = f"{DATA400_PATH}train_{fam}.csv"
    valid_file = f"{DATA400_PATH}valid_{fam}.csv"
    out_path = f"{SAVE400_PATH}{fam}/"
    weigths_path = f"{MODELS400_PATH}{fam}/weights.pmt"

    train = f"sincFold -d cuda train {train_file} --valid-file {valid_file} -o {out_path} -n 10 -w {weigths_path}"
    print(f"Running: {train}")
    os.system(train)
