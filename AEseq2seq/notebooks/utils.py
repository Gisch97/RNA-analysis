import pandas as pd
from mlflow.tracking import MlflowClient

NOTEBOOK_URI="sqlite:////home/guillermo/Documents/SINC/Research/RNA/AEseq2seq/mlruns.db"
SINC_URI="sqlite:////home/gkulemeyer/Documents/Repos/AEseq2seq/mlruns.db"

METRICS = [ 
    "train_loss",
    "valid_loss",
    "test_loss",
    "valid_F1",
    "train_F1",
    "test_F1",
    "train_Accuracy",
    "valid_Accuracy",
    "test_Accuracy",
    "train_Accuracy_seq",
    "valid_Accuracy_seq",
    "test_Accuracy_seq",
]

HYPERPARAMETERS = [
    "hyp_device",
    "hyp_lr",
    "hyp_scheduler",
    "hyp_verbose",
    "hyp_output_th",
    "hyp_test_noise",
    'max_epochs',
]

ARCHITECTURE = [
    "arc_embedding_dim",
    "arc_features",
    "arc_encoder_blocks",
    # "arc_num_params",
    "arc_initial_volume",
    "arc_latent_volume",
    "arc_num_conv",
    "arc_pool_mode",
    "arc_up_mode",
    "arc_addition",
    "arc_skip",
]

DATASETS = ['exp', 'run', 'command', "train_file", "valid_file", "test_file", 'out_path','valid_split']


OTHERS =[
       'no_cache',
       'device',
       'batch_size', 
       'max_len',
       'verbose',
       'cache_path',
       'nworkers',
        ]


def get_params_and_metrics(client, id, columns=None):
    # ID del experimento
    experiment_id = id  # Cambia por el que necesitas

    # Obtener todos los runs del experimento
    runs = client.search_runs(experiment_ids=[experiment_id])

    # Crear un diccionario para agrupar runs por nombre
    grouped_runs = {}

    for run in runs:
        if run.info.lifecycle_stage != "active" and run.info.status != "FINISHED":
            pass
        run_name = run.data.tags.get("mlflow.runName", "Unnamed Run")
        if run_name not in grouped_runs:
            grouped_runs[run_name] = {}

        # Guardar información del run
        grouped_runs[run_name][run.info.run_id] = {
            "run_id": run.info.run_id,
            "params": run.data.params,
            "metrics": run.data.metrics,
        }

    # Crear lista para construir el DataFrame
    data = []

    for run_name, runs_dict in grouped_runs.items():
        combined_data = {"run_name": run_name}

        for run_id, run_data in runs_dict.items():
            # Añadir hiperparámetros
            for param, value in run_data["params"].items():
                combined_data[param] = value

            # Añadir métricas diferenciando train, valid y test
            for metric, value in run_data["metrics"].items():
                if "train" in metric:
                    combined_data[metric] = value
                elif "valid" in metric:
                    combined_data[metric] = value
                elif "test" in metric:
                    combined_data[metric] = value
                else:
                    combined_data[metric] = value  # En caso de que no esté etiquetada

        data.append(combined_data)

    # Crear DataFrame
    df = pd.DataFrame(data)
    # # Rellenar NaN con un valor neutro (opcional)
    # df.fillna("-", inplace=True)
    for file in ["train_file", "valid_file", "test_file"]:
        try:
            df[file] = df[file].str.split("/").str[-1]
        except:
            print(f'falta {file}')
    if columns is not None:
        try:
            df = df[['run_name']+columns]
        except KeyError:
            raise KeyError("Hay columnas que no están en df: ", columns, " intente con las siguientes columnas: ", df.columns)
    return df