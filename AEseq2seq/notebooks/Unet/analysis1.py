# %% [markdown]
"""
# Análisis Exploratorio Reducido y de Sanidad de Datos en Modelos UNet

En este notebook se realiza:
1. Un **análisis de sanidad** de los datos, verificando duplicados, valores faltantes y distribuciones generales.
2. Un **análisis reducido por versión** para comparar, por ejemplo, los experimentos de v1 con compresión vía *strides* vs *avg pooling*,
   diferenciando además por los parámetros `arc_num_conv1` y `arc_num_conv2`.

Se definen las rutas:
- BASE_PATH: `/home/guillermo/Documents/SINC/RNA/analysis/AEseq2seq/`
- DATA_PATH: `notebooks/from_db/Unet/`
"""
# %%
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo para los gráficos
sns.set(style="whitegrid", context="notebook")
# %% [markdown]
"""
## Definición de Rutas

Se definen la ruta base y la ruta relativa donde se encuentran los datos.
"""
# %%
BASE_PATH = '/home/guillermo/Documents/SINC/RNA/analysis/AEseq2seq/'
DATA_PATH = 'notebooks/from_db/Unet/'

# Ruta completa de los datos
data_dir = os.path.join(BASE_PATH, DATA_PATH)
print("Data directory:", data_dir)
# %% [markdown]
"""
## Función para Cargar los Datos

La función `load_data` busca archivos CSV (que contengan "params_best_epoch") de forma recursiva en `data_dir`
y extrae de la ruta:
- La **versión** (p.ej. v1, v2, v3).
- El **subfolder** (que indica la implementación).
- El método de **compresión**:
  - `"avg pooling"` para `pooling_layers`.
  - `"strides"` para `convolutional_layers`, `kernel_strides` o `unet- convolutional-layers`.
"""
# %%
def load_data(path_pattern):
    """
    Busca y carga de forma recursiva todos los archivos CSV que coincidan con el patrón.
    Agrega columnas extraídas de la ruta: version, subfolder y el método de compresión.
    """
    csv_files = glob.glob(path_pattern, recursive=True)
    list_dfs = []
    if not csv_files:
        print("No se encontraron archivos con el patrón:", path_pattern)
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Extraer información de la ruta. Suponemos estructura:
            # BASE_PATH / DATA_PATH / version / subfolder / archivo.csv
            parts = file.split(os.sep)
            version = parts[-3] if len(parts) >= 3 else "unknown"
            subfolder = parts[-2] if len(parts) >= 2 else ""
            
            # Asignar el método de compresión según el nombre del subdirectorio
            subfolder_lower = subfolder.lower()
            if subfolder_lower == "pooling_layers":
                compression = "avg_pooling"
            elif subfolder_lower == "convolutional_layers":
                compression = "strides"
            elif subfolder_lower in ["kernel_strides", "unet- convolutional-layers"]:
                compression = "strides_others"
            else:
                compression = "unknown"
            
            df["version"] = version
            df["subfolder"] = subfolder
            df["compression"] = compression
            df["file_source"] = os.path.basename(file)
            list_dfs.append(df)
        except Exception as e:
            print(f"Error al cargar {file}: {e}")
    return pd.concat(list_dfs, ignore_index=True) if list_dfs else pd.DataFrame()
# %% [markdown]
"""
## Carga de Datos

Se define el patrón de búsqueda utilizando `data_dir` y se carga el DataFrame con los archivos que 
contengan `params_best_epoch` (incluyendo variantes).
"""
# %%
# Definir patrón para buscar archivos de interés en la ruta completa
data_path_pattern = os.path.join(data_dir, "**", "params_best_epoch*.csv")
df = load_data(data_path_pattern)

if df.empty:
    print("No se han cargado datos. Verifica que la ruta y los archivos sean correctos.")
else:
    print("Datos cargados correctamente. Cantidad de registros:", df.shape[0])
# %% [markdown]
"""
## Selección y Conversión de Columnas de Interés

Se definen las columnas de interés (tanto parámetros de arquitectura como métricas de rendimiento)
y se convierten a formato numérico si es necesario.
"""
# %%
columns_of_interest = [
    'name','arc_filters', 'arc_rank', 'arc_num_conv1', 'arc_num_conv2',
    'arc_latent_dim', 'arc_skip_conn','train_loss', 'train_Accuracy', 'train_Accuracy_seq','valid_Accuracy', 'valid_Accuracy_seq',
    'test_loss', 'test_Accuracy', 'test_Accuracy_seq'
]

# Verificar qué columnas están disponibles
available_cols = [col for col in columns_of_interest if col in df.columns]
missing_cols = set(columns_of_interest) - set(available_cols)
if missing_cols:
    print("Las siguientes columnas no se encontraron en los datos:", missing_cols)

# Crear un DataFrame con las columnas de interés y las columnas adicionales
data = df[available_cols + ['version', 'subfolder', 'compression', 'file_source']].copy()

# Convertir las columnas a formato numérico (si es necesario)
for col in available_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
# %% [markdown]
"""
## Análisis de Sanidad de los Datos

En esta sección se revisa la "salud" de los datos, verificando:
- Dimensiones generales y duplicados.
- Valores faltantes por columna.
- Distribución de las variables numéricas.
- Distribución de los métodos de compresión.
"""
# %%
# Información general
print("Dimensiones del DataFrame:", data.shape)
print("Número de filas duplicadas:", data.duplicated().sum())

# Resumen de valores faltantes por columna
missing_summary = data.isnull().sum()
print("\nValores faltantes por columna:")
display(missing_summary)
# %%
# Distribución de los métodos de compresión
plt.figure(figsize=(6,4))
sns.countplot(x="compression", data=data)
plt.title("Cantidad de experimentos por método de compresión")
plt.xlabel("Método de compresión")
plt.ylabel("Conteo")
plt.tight_layout()
plt.show()
# %%
# Histograma de las variables numéricas de interés
data[available_cols].hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.show()
# %% [markdown]
"""
## Análisis Reducido por Versión: Ejemplo para v1

Separamos los experimentos de la versión **v1** y comparamos:
- Experimentos con compresión vía *strides* (convoluciones).
- Experimentos con compresión vía *avg pooling*.

En cada caso se generan gráficos (barplots y boxplots) para visualizar la distribución de `test_Accuracy`
diferenciados por parámetros clave (por ejemplo, `arc_num_conv1` y `arc_num_conv2`).
"""
# %%
# Filtrar datos para la versión v1
data_v1 = data[data["version"] == "v1"]

# Separar según el método de compresión
data_v1_conv = data_v1[data_v1["compression"] == "strides"]
data_v1_pool = data_v1[data_v1["compression"] == "avg pooling"]

print("Experimentos v1 - Strides:", data_v1_conv.shape[0])
print("Experimentos v1 - Avg Pooling:", data_v1_pool.shape[0])
# %% [markdown]
"""
### Barplot: Test Accuracy ordenado por nombre (con hue = arc_num_conv1)

Se ordenan los experimentos de v1 y se muestra la métrica `test_Accuracy` diferenciando por `arc_num_conv1`.
"""
# %%
plt.figure(figsize=(10, 6))
data_v1_sorted = data_v1.sort_values(by="test_Accuracy")
# Usar 'name' si existe; de lo contrario, 'run_uuid'
y_col = 'name' if 'name' in data_v1_sorted.columns else 'run_uuid'
g = sns.barplot(data=data_v1_sorted, x="test_Accuracy", y=y_col, hue="arc_num_conv1")
g.set_title("v1: Test Accuracy (hue: arc_num_conv1)")
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.show()
# %% [markdown]
"""
### Boxplot: Test Accuracy por Método de Compresión y diferenciación por arc_num_conv2

Se muestran boxplots que permiten comparar la distribución de `test_Accuracy` entre experimentos
con compresión por *strides* y *avg pooling*, diferenciando además por `arc_num_conv2`.
"""
# %%
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_v1, x="compression", y="test_Accuracy", hue="arc_num_conv2")
plt.title("v1: Test Accuracy por método de compresión (hue: arc_num_conv2)")
plt.xlabel("Método de compresión")
plt.ylabel("Test Accuracy")
plt.tight_layout()
plt.show()

# %%
