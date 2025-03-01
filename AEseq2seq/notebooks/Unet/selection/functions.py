import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os


def read_params_and_metrics(BASE_PATH, DATA_PATH, columns=None):
    # Cargar los archivos CSV
    data = pd.read_csv(os.path.join(BASE_PATH, DATA_PATH, "params_best_epoch.csv"))
    metrics = pd.read_csv(os.path.join(BASE_PATH, DATA_PATH, "train_metrics.csv"))

    # Verificar si data tiene la columna 'name'
    if "name" not in data.columns:
        warnings.warn("La columna 'name' no está presente en data.", UserWarning)

    # Verificar si columns está definido
    if columns:
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            raise KeyError(f"Las siguientes columnas no están en data: {missing_cols}")
        data = data[columns]
    
    # Eliminar columnas específicas antes del merge
    drop_cols = ["best_epoch", "train_Accuracy", "valid_Accuracy", "test_Accuracy"]
    cols = [col for col in data.columns if col not in drop_cols]
    
    # Fusionar `metrics` con `data`
    if "run_uuid" in metrics.columns and "run_uuid" in data.columns:
        metrics = pd.merge(
            metrics,
            data[cols],
            how="left",
            on="run_uuid",
        )
    else:
        raise KeyError("La columna 'run_uuid' no está presente en ambas tablas.")

    return data, metrics



def graficar_correlacion(df, columnas, titulo):
    """
    Genera y muestra una matriz de correlación a partir de un DataFrame.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame que contiene los datos a analizar.
    columnas : list
        Lista de nombres de columnas sobre las cuales calcular la correlación.
    titulo : str
        Título de la gráfica.

    Retorna:
    --------
    None
        Muestra el heatmap de correlación pero no retorna ningún valor.

    Ejemplo de uso:
    ---------------
    cols_arq = ["arc_filters", "arc_rank", "arc_num_conv1", "arc_num_conv2", "arc_latent_dim"]
    cols_vol = ["L_1", "vol_1", "L_2", "vol_2", "vol_eff", "vol_ratio", "latent_ratio"]
    cols_perf = ["train_Accuracy", "valid_Accuracy", "test_Accuracy"]
    
    graficar_correlacion(data, cols_arq + cols_vol + cols_perf, "Matriz de Correlación General")
    """
    # Verificar si el DataFrame está vacío
    if df.empty:
        warnings.warn(f"El DataFrame proporcionado para {titulo} está vacío.", UserWarning)
        return
    
    # Verificar si todas las columnas existen en el DataFrame
    columnas_faltantes = [col for col in columnas if col not in df.columns]
    if columnas_faltantes:
        warnings.warn(f"Las siguientes columnas no están en el DataFrame: {columnas_faltantes}", UserWarning)
        return

    # Calcular la matriz de correlación y graficarla
    corr = df[columnas].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(titulo)
    plt.show()
