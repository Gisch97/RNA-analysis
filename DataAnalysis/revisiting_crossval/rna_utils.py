# rna_struct_utils.py
# ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations_with_replacement
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math


# ╭───────────────────────── 1. CARGA Y LIMPIEZA ─────────────────────────╮
def load_and_align(corr_path: str, meta_path: str):
    """
    Lee los CSV/TSV de corr (distancias) y meta, y devuelve ambas tablas
    alineadas (solo IDs comunes y mismo orden).

    corr_path : ruta a CSV (índice y columnas = IDs, valores = distancia)
    meta_path : ruta a CSV con columna 'id' y 'fam' (+ otras)
    """
    corr = pd.read_hdf(corr_path)
    meta = pd.read_csv(meta_path)

    common = sorted(set(corr.index) & set(meta["id"]))
    corr = corr.loc[common, common]

    meta = meta[meta["id"].isin(common)].set_index("id").loc[common]  # re-orden
    return corr, meta


# ╭────────────────── 2.0 SUB-MATRICES POR FAMILIA INTRA ─────────────────────╮
def build_intra_fam_corrs(corr: pd.DataFrame, meta: pd.DataFrame):
    """
    Devuelve dict {fam: sub-matrix de distancias}.
    """
    idx_by_fam = meta.groupby("fam").groups
    return {fam: corr.loc[ids, ids].copy() for fam, ids in idx_by_fam.items()}


# ╭────────────────── 2.1 SUB-MATRICES POR FAMILIA INTER ─────────────────────╮
def build_inter_fam_corrs(corr: pd.DataFrame, meta: pd.DataFrame):
    """
    Devuelve dict {fam: sub-matrix de distancias}.
    """
    inter = {}
    idx_by_fam = meta.groupby("fam").groups
    for f1, id1 in idx_by_fam.items():
        inter[f1] = {}
        for f2, id2 in idx_by_fam.items():
            inter[f1][f2] = corr.loc[id1, id2].copy()
    return inter


# ╭────────────────── 2.2 STR POR FAMILIA ─────────────────────╮
def table_inter_fam(corrs: dict):
    strength = {}
    for f, dic in corrs.items():
        strength[f] = {
            fam: mat.values[np.triu_indices_from(mat, k=1)].mean()
            for fam, mat in dic.items()
        }
    return strength


# ╭────────────────── 2.3 MAX POR FAMILIA ─────────────────────╮
def table_max_inter_fam(corrs: dict):
    strength = {}
    for f, dic in corrs.items():
        strength[f] = {
            fam: mat.values[np.triu_indices_from(mat, k=1)].max()
            for fam, mat in dic.items()
        }
    return strength


# ╭────────────────── 2.3 MIN POR FAMILIA ─────────────────────╮
def table_min_inter_fam(corrs: dict):
    strength = {}
    for f, dic in corrs.items():
        strength[f] = {
            fam: mat.values[np.triu_indices_from(mat, k=1)].min()
            for fam, mat in dic.items()
        }
    return strength


# ╭──────────── 3. FILTRO POR UMBRAL Y TAMAÑO DE CLÚSTER ──────────────╮
def filter_by_threshold(corrs: dict, threshold: float, min_family_size: int):
    """
    Para cada familia en `corrs`, elimina secuencias demasiado cercanas (distancia <= threshold),
    pero SI la familia tiene menos de `min_family_size` secuencias, se deja intacta.

    Parámetros
    ----------
    corrs : dict
        {'fam': DataFrame}, cada DataFrame es una matriz de distancias
        índices = columnas = IDs.
    threshold : float
        Distancia máxima permitida; si d(i,j) <= threshold, se consideran “demasiado similares”.
    min_family_size : int
        Tamaño mínimo de familia para aplicar el filtro; familias más pequeñas se mantienen completas.

    Retorna
    -------
    corrs_filt : dict
        {'fam': DataFrame filtrada}, cada sub-matriz con secuencias “independientes” o
        intacta si la familia era pequeña.
    """
    corrs_filt = {}

    for fam, mat in corrs.items():
        ids = list(mat.index)
        n = len(ids)

        # Si la familia es pequeña, la dejamos completa
        if n < min_family_size:
            corrs_filt[fam] = mat.copy()
            continue

        # 1. ÍNDICES triángulo superior sin diagonal
        i_idx, j_idx = np.triu_indices(n, k=1)
        vals = mat.values[i_idx, j_idx]

        # 2. Filtrar parejas demasiado cercanas
        mask_close = vals >= threshold
        i_close = i_idx[mask_close]
        j_close = j_idx[mask_close]
        d_close = vals[mask_close]

        # 3. Si no hay pares, mantenemos familia completa
        if d_close.size == 0:
            corrs_filt[fam] = mat.copy()
            continue

        # 4. Ordenar parejas por distancia ascendente
        order = np.argsort(d_close)
        i_close = i_close[order]
        j_close = j_close[order]

        # 5. Array booleano para marcar qué secuencias mantener
        keep = np.ones(n, dtype=bool)

        # 6. Recorrer parejas y desactivar la segunda si ambas están activas
        for a, b in zip(i_close, j_close):
            if keep[a] and keep[b]:
                keep[b] = False

        # 7. Construir la sub-matriz con IDs marcados como keep=True
        kept_ids = [ids[idx] for idx in np.nonzero(keep)[0]]
        corrs_filt[fam] = mat.loc[kept_ids, kept_ids].copy()

    return corrs_filt


# ╭──────────── 3.1 FILTRO POR PORCENTAJE Y TAMAÑO DEL DATASET SEGUN REPETICIONES ──────────────╮
def filter_by_count(data, data_size=0, percentile=0.5, verbose=False):
    M = data.copy()
    np.fill_diagonal(M.values, 0)
    print(f"Initial shape: {M.shape[0]}^2")
    if (data_size != 0) & (M.shape[0] < data_size):
        print(
            f"Data size ({data_size}) is larger than matrix size ({M.shape[0]}), ignoring family."
        )
        return M
    # Calcular el percentil
    if data_size == 0:
        thr = int(percentile * M.shape[0])
        print(
            f"Threshold for dropping:  ({str(100*percentile)}% of {M.shape[0]} - target shape: {thr})"
        )
    if data_size > 0:
        thr = data_size
        print(
            f"Data size set to {data_size}, ignoring percentile. Target percentile: {100*data_size/M.shape[0]}"
        )

    max_val = M.values.max()
    while M.shape[0] > thr:
        max_val = M.values.max()
        # Contar cuántas veces aparece max_val en cada fila
        count = (M == max_val).sum(axis=1).values
        drop_idx = np.argmax(count)
        drop_label = M.index[drop_idx]  # obtener etiqueta real del índice
        # Eliminar fila y columna
        M = M.drop(index=drop_label, columns=drop_label)

        if verbose:
            if M.shape[0] % 33 == 0:
                print(
                    f"Shape: {M.shape[0]}, Perc: {round(100*M.shape[0]/data.shape[0],1)}, counter: {count[drop_idx]}, max_val: {round(max_val,4)} | "
                    f"dropped: '{drop_label}',"
                )
    print(f"Final shape: {M.shape}, max_val: {max_val}")
    return M


def filter_by_count_dict(corrs: dict, data_size=0, percentile=0.5, verbose=False):
    """
    Filtra un diccionario de matrices de distancias por un porcentaje o tamaño de dataset.

    Parámetros
    ----------
    corrs : dict
        {'fam': DataFrame}, cada DataFrame es una matriz de distancias.
    data_size : int
        Tamaño objetivo del dataset filtrado (0 para usar percentil).
    percentile : float
        Percentil para determinar el tamaño del dataset (0.5 = 50%).
    verbose : bool
        Si True, imprime información detallada del proceso.

    Retorna
    -------
    dict
        {'fam': DataFrame filtrada}, cada sub-matriz con secuencias filtradas.
    """
    return {
        fam: filter_by_count(mat, data_size, percentile, verbose)
        for fam, mat in corrs.items()
    }


# ╭──────────── 3.1 FILTRO POR PORCENTAJE Y TAMAÑO DEL DATASET SEGUN REPETICIONES ──────────────╮
def filter_struc_by_count(data, data_size=0, percentile=0.5, verbose=False):
    M = data.copy()
    np.fill_diagonal(M.values, 1)
    print(f"Initial shape: {M.shape[0]}^2")
    if (data_size != 0) & (M.shape[0] < data_size):
        print(
            f"Data size ({data_size}) is larger than matrix size ({M.shape[0]}), ignoring family."
        )
        return M
    # Calcular el percentil
    if data_size == 0:
        thr = int(percentile * M.shape[0])
        print(
            f"Threshold for dropping:  ({str(100*percentile)}% of {M.shape[0]} - target shape: {thr})"
        )
    if data_size > 0:
        thr = data_size
        print(
            f"Data size set to {data_size}, ignoring percentile. Target percentile: {100*data_size/M.shape[0]}"
        )

    min_val = M.values.min()
    while M.shape[0] > thr:
        min_val = M.values.min()
        # Contar cuántas veces aparece max_val en cada fila
        count = (M == min_val).sum(axis=1).values
        drop_idx = np.argmax(count)
        drop_label = M.index[drop_idx]  # obtener etiqueta real del índice
        # Eliminar fila y columna
        M = M.drop(index=drop_label, columns=drop_label)

        if verbose:
            if M.shape[0] % 33 == 0:
                print(
                    f"Shape: {M.shape[0]}, Perc: {round(100*M.shape[0]/data.shape[0],1)}, counter: {count[drop_idx]}, min_val: {round(min_val,4)} | "
                    f"dropped: '{drop_label}',"
                )
    print(f"Final shape: {M.shape}, min_val: {min_val}")
    # np.fill_diagonal(M.values, 1)
    return M


def filter_struc_by_count_dict(corrs: dict, data_size=0, percentile=0.5, verbose=False):
    """
    Filtra un diccionario de matrices de distancias por un porcentaje o tamaño de dataset.

    Parámetros
    ----------
    corrs : dict
        {'fam': DataFrame}, cada DataFrame es una matriz de distancias.
    data_size : int
        Tamaño objetivo del dataset filtrado (0 para usar percentil).
    percentile : float
        Percentil para determinar el tamaño del dataset (0.5 = 50%).
    verbose : bool
        Si True, imprime información detallada del proceso.

    Retorna
    -------
    dict
        {'fam': DataFrame filtrada}, cada sub-matriz con secuencias filtradas.
    """
    return {
        fam: filter_struc_by_count(mat, data_size, percentile, verbose)
        for fam, mat in corrs.items()
    }


# ╭───────────── 4. ORDEN DE FAMILIAS Y MATRICES ──────────────────────╮
def family_means(corrs: dict):
    """
    Devuelve dict fam → distancia media (sin diagonal).
    """
    return {
        fam: mat.values[np.triu_indices_from(mat, k=1)].mean()
        for fam, mat in corrs.items()
    }


def order_families(corrs: dict, ascending=True):
    """
    Lista de familias ordenadas por distancia media.
    """
    means = family_means(corrs)
    return sorted(means, key=means.get, reverse=not ascending)


def order_matrix(mat: pd.DataFrame, method="mean"):
    """
    Re-ordena filas/columnas para visualización.
    method = 'mean'  → por distancia media (centrales primero)
           = 'cluster' → clustering jerárquico (average linkage)
    """
    if method == "cluster":
        cond = squareform(mat.values, checks=False)
        Z = linkage(cond, method="average")
        order = leaves_list(Z)
        ids = mat.index[order]
        return mat.loc[ids, ids]

    # default: mean distance
    avg = mat.mean(axis=0)
    ids = avg.sort_values().index
    return mat.loc[ids, ids]


# ╭────────────────── 5. RESUMEN NUMÉRICO RÁPIDO ──────────────────────╮
def summary_table(corrs: dict):
    rows = []
    for fam, mat in corrs.items():
        vals = mat.values[np.triu_indices_from(mat, k=1)]
        rows.append(
            {
                "Fam": fam,
                "n_seq": mat.shape[0],
                "mean_dist": vals.mean() if vals.size else np.nan,
                "min_dist": vals.min() if vals.size else np.nan,
                "max_dist": vals.max() if vals.size else np.nan,
            }
        )
    return (
        pd.DataFrame(rows).sort_values("n_seq", ascending=False).reset_index(drop=True)
    )


def summary_table_inter(corrs: dict):
    rows = []
    for fam, mat in corrs.items():
        vals = mat.values[np.triu_indices_from(mat, k=1)]
        rows.append(
            {
                "Fam": fam,
                "n_seq": mat.shape[0],
                "mean_dist": vals.mean() if vals.size else np.nan,
                "min_dist": vals.min() if vals.size else np.nan,
                "max_dist": vals.max() if vals.size else np.nan,
            }
        )
    return (
        pd.DataFrame(rows).sort_values("n_seq", ascending=False).reset_index(drop=True)
    )


# ╭────────────────── 5.1 RESUMEN Relativo ────────────────────────────╮
def relative_summary_table(corrs, corrs_filtr):
    df = pd.DataFrame()
    df["Fam"] = corrs["Fam"]
    df["%n_seq"] = 100 * corrs_filtr["n_seq"] / corrs["n_seq"]
    df["r_min_variation"] = corrs_filtr["min_dist"] / corrs["min_dist"]
    df["r_mean_variaton"] = corrs_filtr["mean_dist"] / corrs["mean_dist"]
    return df


# ╭───────── 6. HEATMAPS + HISTOGRAMAS EMBEBIDOS (PANEL) ──────────────╮
def plot_heatmaps_with_hist(
    corrs,
    n_fams=9,
    order_method="mean",
    n_cols=3,
    figsize_scale=(4, 3),
):
    """
    Dibuja hasta n_fams heatmaps (ordenados por cohesión) con un histograma
    de distancias bajo cada mapa. La barra de color se comparte.

    corrs        : dict {fam: DataFrame distancias}
    n_fams       : cuántas familias mostrar
    order_method : 'mean' o 'cluster'
    bin_width    : resolución del histograma
    n_cols       : columnas del grid
    figsize_scale: tupla (ancho_por_col, alto_por_fila)
    """
    ordered = order_families(corrs)
    n_rows = math.ceil(n_fams / n_cols)

    fig = plt.figure(
        figsize=(figsize_scale[0] * n_cols + 1.2, figsize_scale[1] * n_rows)
    )
    gs = gridspec.GridSpec(
        n_rows,
        n_cols + 1,
        width_ratios=[1] * n_cols + [0.06],
        wspace=0.05,
        hspace=0.15,
    )

    for idx, fam in enumerate(ordered):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        mat = order_matrix(corrs[fam], method=order_method)
        last_im = ax.imshow(mat.values, aspect="auto", cmap="viridis")
        ax.set_title(f"{fam} (n={mat.shape[0]})", fontsize=13)
        ax.set_xticks([]), ax.set_yticks([])

    # ocultar celdas vacías
    for idx in range(len(ordered), n_rows * n_cols):
        fig.add_subplot(gs[idx // n_cols, idx % n_cols]).axis("off")

    # barra de color
    cax = fig.add_subplot(gs[:, -1])
    fig.colorbar(last_im, cax=cax, label="Distancia")

    fig.suptitle(
        f"Heatmaps de distancias secuenciales por familias",
        y=0.95,
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 0.96, 0.9])
    plt.show()


# ╭───────────── 7. HISTOGRAMA SOLO (MULTI-PANEL) ─────────────────────╮
def plot_histograms(
    corrs, bin_width=0.05, n_cols=3, figsize_scale=(3.5, 2.5), ordered=True
):
    fams = order_families(corrs) if ordered else list(corrs.keys())
    n_rows = math.ceil(len(fams) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_scale[0] * n_cols, figsize_scale[1] * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for ax, fam in zip(axes.ravel(), fams):
        vals = corrs[fam].values[np.triu_indices_from(corrs[fam], k=1)]
        bins = np.arange(0, vals.max() + bin_width, bin_width) if vals.size else [0, 1]
        ax.hist(vals, bins=bins, density=True, edgecolor="black", alpha=0.85)
        ax.set_title(f"{fam} (n = {corrs[fam].shape[0]})", fontsize=13)
        ax.grid()
    for ax in axes.ravel()[len(fams) :]:
        ax.axis("off")
    axes[0, 0].set_ylabel("Cuentas normalizadas")
    axes[1, 0].set_ylabel("Cuentas normalizadas")
    axes[2, 0].set_ylabel("Cuentas normalizadas")
    axes[2, 0].set_xlabel("Distancia")
    axes[2, 1].set_xlabel("Distancia")
    axes[2, 2].set_xlabel("Distancia")
    fig.suptitle("Distribución de pesos por familia", y=0.97)
    plt.tight_layout()
    plt.show()
