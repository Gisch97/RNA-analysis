import pandas as pd
import os
import ast
from utils import log2pd, dot2bp, f1_score


def extract_info(path, fam, source, ep):
    # for f in fams: for ep in eps: ...
    EPOCHS = {"ep_015": 15, "ep_050": 50, "ep_100": 100, "ep_150": 150, "ep_200": 200}
    df = log2pd(path)
    df["base_pairs"] = df["structure"].apply(dot2bp)

    df["test_f1"] = df.apply(
        lambda row: f1_score(
            df_ref.loc[row.id, "base_pairs"],  # lista de pares ref
            row["base_pairs"],  # lista de pares pred
        ),
        axis=1,
    )
    return {
        "fam": fam,
        "source": source,
        "epoch": EPOCHS[ep],
        "test_f1": df.test_f1.mean(),
    }
    

DATA_REF_PATH = "/home/gkulemeyer/Documents/Repos/RNA-analysis/DataAnalysis/data/sources/ArchiveII.csv"
df_ref = pd.read_csv(DATA_REF_PATH, index_col="id")
df_ref["base_pairs"] = df_ref["base_pairs"].apply(ast.literal_eval)



sources = ['hc_100',
    'rnadist_100',
    'samples_100',
    'hc_200',
    'rnadist_200',  
    'samples_200',
    'hc_400',
    'rnadist_400',
    'samples_400',]

for source in sources:
    DATA_ROOT = f"../BRONZE/logs/ArchiveII_{source}/" 
    SAVE_PATH = f"../BRONZE/f1_by_epoch/ArchiveII_{source}/"

    fams = os.listdir(DATA_ROOT)
    fams.sort()
    eps = os.listdir(DATA_ROOT + fams[0])
    eps.sort()


    
    rows = []
    for f in fams:
        for ep in eps:
            test_log = DATA_ROOT + f + "/" + ep + "/test.log"
            row = extract_info(test_log, f, source, ep)
            rows.append(row)
    table = pd.DataFrame(rows)


    os.makedirs(SAVE_PATH, exist_ok=True)
    table.to_csv(SAVE_PATH + "test_f1.csv", index=False)