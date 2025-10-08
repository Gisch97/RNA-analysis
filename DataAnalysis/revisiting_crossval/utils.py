import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import numpy as np
import pandas as pd

######## EXTRACT DATA ########

def get_ids(dist, split, fold, partition):
    """
    Obtiene los ids comunes entre la matriz de distancias y una particion dada
    """
    f = split.query("fold_number == @fold")
    part = f.query("partition == @partition")
    return sorted(set(dist.index) & set(part["id"]))


def get_partition_distances(dist, split, fold, partition1, partition2):
    """
    Obtiene la matriz de distancias entre dos particiones
    rows: ids de la primera particion
    cols: ids de la segunda particion
    """
    rows = get_ids(dist, split, fold, partition1)
    cols = get_ids(dist, split, fold, partition2)
    return dist.loc[rows, cols].copy()


######## BUILD DFS ########


def create_dfs(splits, verbose=False):
    # train = train + valid  
    DIST_PATH = "../data/rnadist_f_all.h5"
    dist = pd.read_hdf(DIST_PATH)
    splits["partition"] = splits["partition"].apply(
    lambda x: "train" if x in ["train", "valid"] else "test"
    )
    folds = splits.fold_number.unique()
    parts = list(splits.partition.unique())
    
    dfs = {fold : {s: {p: pd.DataFrame() for p in parts} for s in parts} for fold in folds}
    for fold in folds:
        for p1 in parts:
            for p2 in parts:
                dfs[fold][p1][p2] = get_partition_distances(dist, splits, fold, p1, p2)
                if verbose:
                    print(f"Fold {fold} - {p1} vs {p2}: {dfs[fold][p1][p2].shape}")
    return dfs


def create_stats(dfs, folds):
    stats = {"fold": [], "min": [],"mean": [],"max": [] }
    for fold in folds:
        A = dfs[fold]["train"]["test"].values
        stats["fold"].append(fold)
        stats["min"].append(np.nanmin(A))
        stats["mean"].append(np.nanmean(A))
        stats["max"].append(np.nanmax(A))
    df_stats = pd.DataFrame(stats)
    return df_stats
 
 
def create_flat_dfs(dfs, folds):
    dfs_dists = {}
    for fold in folds:
        rows = []
        mat = dfs[fold]['train']['test']
        vals = mat.to_numpy().ravel()
        rows.extend({ 
            "dist": float(v)
        } for v in vals)

        dfs_dists[fold] = pd.DataFrame(rows)
    return dfs_dists
 
 
 ######## MAKE PLOTS ########
 
 
def plot_heatmap_stats(df_stats,index, save_path, SAVE=False):
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        df_stats.set_index(index),
        annot=True, fmt=".3f",
        cmap="viridis",
        vmin=0, vmax=1.41
    )
    plt.ylabel("fold")
    plt.title("Structural distance train/test")
    if SAVE:
        plt.savefig(save_path+"heatmap_stats.pdf")
    plt.show() 
 
 
def plot_heatmap_ids(dfs, save_path, SAVE=False):

    fig, axes = plt.subplots(1, 5, figsize=(18, 6), constrained_layout=True)
    vmin, vmax = 0, 1.41  # escala fija
    for fold in range(5):
        mat = dfs[fold]['train']['test'].values
        im = axes[fold].imshow(mat, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        axes[fold].set_title(f"Fold {fold}")
        axes[fold].axis('off')  # quita los ticks/labels

    # agregar barra de color única
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
    cbar.set_label("Distance")

    plt.suptitle("Structural distance train/test by id", fontsize=16)
    if SAVE:
        plt.savefig(save_path+"heatmap_ids.pdf")
    plt.show()


def plot_heatmap_idsxfam(dfs, splits, save_path, SAVE=False):

    # asignar familia a cada id
    splits["fam"] = splits["id"].str.split("_").str[0]
    id2fam = dict(zip(splits["id"], splits["fam"]))

    # paleta de colores (Set3 es buena para categorías)
    fams = sorted(set(id2fam.values()))
    cmap = plt.cm.get_cmap("tab10", len(fams))
    fam2color = {fam: cmap(i) for i, fam in enumerate(fams)}

    fig, axes = plt.subplots(1, len(dfs), figsize=(18, 6), constrained_layout=True)

    vmin, vmax = 0, 1.41
    for fold in range(len(dfs)):
        mat = dfs[fold]['train']['test']  # dataframe con index/columns = ids
        im = axes[fold].imshow(mat.values, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        axes[fold].set_title(f"Fold {fold}")
        axes[fold].axis("off")

        # familias filas y columnas → colores
        row_colors = [fam2color[id2fam[i]] for i in mat.index]
        col_colors = [fam2color[id2fam[i]] for i in mat.columns]

        # barrita arriba (columnas)
        ax_top = inset_axes(axes[fold], width="100%", height="0.5%", loc="upper center",
                        bbox_to_anchor=(0, 0, 1, 1), bbox_transform=axes[fold].transAxes, borderpad=-0.4)
        ax_top.imshow([col_colors], aspect="auto")
        ax_top.axis("off")

        # barrita izquierda (filas)
        ax_left = inset_axes(axes[fold], width="3%", height="100%", loc="center left",
                            bbox_to_anchor=(0, 0, 1, 1), bbox_transform=axes[fold].transAxes, borderpad=0)
        ax_left.imshow(np.array(row_colors)[:, None].reshape(-1, 1, 4), aspect="auto")
        ax_left.axis("off")

    # colorbar única para el heatmap
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
    cbar.set_label("Distance")

    # leyenda de familias
    handles = [plt.Line2D([0], [0], color=fam2color[f], lw=6) for f in fams]
    # leyenda abajo, centrada, con varias columnas
    fig.legend(handles, fams, title="Families", bbox_to_anchor=(0.5, -0.08), loc='lower center', ncol=len(fams), frameon=False, fontsize=10)

    plt.suptitle("Structural distance train/test by families", fontsize=16)
    if SAVE:
        plt.savefig(save_path+"heatmap_ids_fam.pdf")
    plt.show()

    
def plot_hist_subplots(dfs_dists, folds, save_path, SAVE=False):
    n_folds = len(folds)
    cmap = cm.get_cmap('viridis', n_folds)
    colors = [cmap(i) for i in range(n_folds)]

    fig, axes = plt.subplots(1, n_folds, figsize=(20, 4), constrained_layout=True, sharey=True, sharex=True)

    for i, fold in enumerate(folds):
        df = dfs_dists[fold]
        axes[i].hist(
            df['dist'],
            bins=50,
            density=True,
            histtype='step',
            color=colors[i],
            linewidth=1.5
        )
        axes[i].set_title(f"Fold {fold}")
        axes[i].set_xlabel("Distance")
        axes[i].set_ylabel("")
        axes[i].grid(True, ls="--", alpha=0.5)

    axes[0].set_xlabel("Density")
    plt.suptitle("Train/test distance distribution by fold", fontsize=16)
    plt.tight_layout()
    if SAVE:
        plt.savefig(save_path+"hist_subplots.pdf")
    plt.show()

    
def plot_hist(dfs_dists, folds, save_path, SAVE=False):
    plt.figure(figsize=(10, 6))
    # Genero colores de la paleta viridis
    cmap = cm.get_cmap('viridis', len(folds))
    colors = [cmap(i) for i in range(len(folds))]

    for i, fold in enumerate(folds):
        df = dfs_dists[fold]
        plt.hist(
            df['dist'],
            bins=50,
            density=True,
            histtype='step',
            color=colors[i],
            linewidth=1.5,
            label=f"{fold}"
        )

    plt.title("Structural distance distribution train/test by fold")
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(title="Fold")
    if SAVE:
        plt.savefig(save_path+"hist.pdf")
    plt.show()    

    
def plot_coverage(dfs, folds, save_path, SAVE=False):
    thresholds = np.linspace(0, 1.4, 50)
    cmap = cm.get_cmap('viridis', len(folds))
    colors = [cmap(i) for i in range(len(folds))]

    plt.figure(figsize=(8, 6))
    cover_metric = {}
    for i, fold in enumerate(folds):
        mat = dfs[fold]['train']['test']
        vals_dist = mat.values.ravel()
        cover_metric[fold] = (vals_dist <= 0.5).mean().item()
        coverage = [(vals_dist <= t).mean() for t in thresholds]
        plt.plot(
            thresholds,
            coverage,
            label=f"{fold}",
            color=colors[i]
        )
    plt.title("Coverage vs Threshold (train/test)")
    plt.xlabel("Threshold")
    plt.ylabel("Coverage")
    plt.grid(True, ls="--", alpha=0.6)
    plt.ylim(0, 1)
    plt.legend(title="Fold")
    if SAVE:
        plt.savefig(save_path+"coverage.pdf")
    plt.show() 
    