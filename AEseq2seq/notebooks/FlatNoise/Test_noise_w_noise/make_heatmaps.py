import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




df = pd.read_csv("metrics.csv") 
# Asegurar que los datos estén bien tipados
df["train_swaps"] = df["train_swaps"].astype(float)
df["test_swaps"] = df["test_swaps"].astype(float)
df["test_Accuracy"] = df["test_Accuracy"].astype(float)

# Obtener todos los run_names únicos
run_names = df['name'].unique()

# Crear un heatmap por cada run_name
for run in run_names:
    sub_df = df[df['name'] == run]

    # Agrupar por train_swaps y test_swaps, y promediar test_Accuracy
    heatmap_data = sub_df.groupby(['train_swaps', 'test_swaps'])['test_Accuracy'].mean().reset_index()

    # Crear matriz para heatmap
    pivot_table = heatmap_data.pivot(index='train_swaps', columns='test_swaps', values='test_Accuracy')
    pivot_table = pivot_table.sort_index(axis=0, ascending=False).sort_index(axis=1, ascending=True)

    # Crear figura
    plt.figure(figsize=(11, 7))
    sns.heatmap(
        pivot_table,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        vmin=0.25,  
        vmax=1.0,    
        linewidths=0.5,
        linecolor='gray',
        cbar=True
    )

    plt.title(f"Test Accuracy Heatmap - {run}", fontsize=14)
    plt.xlabel("Test Swaps", fontsize=12)
    plt.ylabel("Train Swaps", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'heatmaps/{run}heatmap.png')
    plt.close()
