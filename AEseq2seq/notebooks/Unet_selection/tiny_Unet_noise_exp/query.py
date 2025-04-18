# n?

import pandas as pd
import os
from mlflow.tracking import MlflowClient


from utils import get_params_and_metrics
from utils import METRICS, HYPERPARAMETERS, ARCHITECTURE, DATASETS, OTHERS

NOTEBOOK_URI="sqlite:////home/guillermo/Documents/SINC/Research/RNA/AEseq2seq/mlruns.db"
SINC_URI="sqlite:////home/gkulemeyer/Documents/Repos/AEseq2seq/mlruns.db"


# Conectar al backend store
client = MlflowClient(tracking_uri=SINC_URI)
experiments = client.search_experiments()
for i,exp in enumerate(experiments):
    print(f"i: {i}, ID: {exp.experiment_id}, Nombre: {exp.name}, Estado: {exp.lifecycle_stage}")

################    
n = XXX
################

name = experiments[n].name
id = experiments[n].experiment_id

print(f"Nombre: {name}, ID: {id}, Estado: {experiments[n].lifecycle_stage}")

df_arc = get_params_and_metrics(client, id, ARCHITECTURE)
for col in ARCHITECTURE:
    print(f"{col}: {df_arc[col].unique()}")
    
# METRICS, HYPERPARAMETERS, ARCHITECTURE, DATASETS, OTHERS
cols = [ 
    "arc_features",
    "arc_skip",
    "arc_num_conv",
    "n_swaps"
]

df_metrics = get_params_and_metrics(client, id,  cols + METRICS)

EXPERIMENT_PATH = os.path.join("Unet_selection", name)
os.makedirs(EXPERIMENT_PATH, exist_ok=True)
df_metrics.to_csv(os.path.join(EXPERIMENT_PATH, "metrics.csv"), index=False)