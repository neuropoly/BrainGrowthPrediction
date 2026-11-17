#!/usr/bin/env python3
# scripts/analyze/full_pipeline.py
"""
Pipeline complet (converti depuis le notebook round_latent_emporte_piece_pipeline-5-aout.ipynb) :

  1. Chargement des latents (.npz) references dans un CSV
  2. Arrondi des valeurs des latents (1 canal ou fusion de plusieurs canaux)
  3. "Emporte-piece" : suppression du fond / des coins (masque de cerveau) - optionnel
  4. Standardisation - optionnelle
  5. Reduction de dimension : PCA / TruncatedSVD / DictionaryLearning / SparsePCA
  6. Clustering : KMeans / SpectralClustering / GaussianMixture / CAH / HDBSCAN
  7. Sauvegarde : figures (.png), embeddings (.npy), modeles (.joblib), csv avec labels

Utilisation (defauts = comportement d'origine du notebook) :
    python scripts/analyze/full_pipeline.py--dataset_csv data/dataset338.csv

Exemple avec les nouvelles options :
    python scripts/analyze/full_pipeline.py\
        --dataset_csv data/dataset338.csv \
        --channel fusion \
        --reduction_method truncatedsvd --n_components 10 \
        --clustering_method gmm \
        --no-standardize \
        --no-mask

Tous les parametres sont modifiables en argument de ligne de commande (voir --help).
"""

import argparse
import os
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, TruncatedSVD, DictionaryLearning, SparsePCA
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


REDUCTION_METHODS = ["pca", "truncatedsvd", "dictionarylearning", "sparsepca"]
CLUSTERING_METHODS = ["kmeans", "spectral", "gmm", "cah", "hdbscan"]


# ---------------------------------------------------------------------------
# 1. Chargement des donnees
# ---------------------------------------------------------------------------

def load_latents(dataset_csv):
    """Charge le csv + les latents .npz associes (colonne 'latent_path')."""
    df = pd.read_csv(dataset_csv)

    latents_raw = []
    kept_rows = []

    print("Loading latents data...")
    for index, row in df.iterrows():
        latent_path = row["latent_path"]
        if os.path.exists(latent_path):
            data = np.load(latent_path)
            latents_raw.append(data["latent"])
            kept_rows.append(index)
        else:
            print(f"Latent manquant pour {row.get('subject_id', index)}")

    df = df.loc[kept_rows].reset_index(drop=True)
    latents = np.squeeze(np.array(latents_raw), axis=1)
    print(f"Loaded.\nLatent initial shape: {latents.shape[0]} sujets, {latents.shape}")
    return df, latents


def select_channels(latents, channel_arg):
    """
    channel_arg : "fusion" -> tous les canaux, sinon un entier (un seul canal).
    Retourne (X, channels_used) avec X de shape (N, C, D, H, W).
    """
    n_channels_available = latents.shape[1]

    if str(channel_arg).lower() == "fusion":
        channels = list(range(n_channels_available))
    else:
        channels = [int(channel_arg)]

    X = latents[:, channels]
    print(f"Canaux utilises : {channels} -> shape {X.shape}")
    return X, channels


# ---------------------------------------------------------------------------
# 2. Visualisation (coupes axiale / coronale / sagittale)
# ---------------------------------------------------------------------------

def plot_latent_alone(volume, title, save_path):
    D, H, W = volume.shape
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(volume[:, :, W // 2], cmap="gray")
    ax[0].set_title("Axiale")
    ax[0].axis("off")

    ax[1].imshow(volume[:, H // 2, :], cmap="gray")
    ax[1].set_title("Coronale")
    ax[1].axis("off")

    ax[2].imshow(volume[D // 2, :, :], cmap="gray")
    ax[2].set_title("Sagittale")
    ax[2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardee : {save_path}")


def plot_latent_with_mask(latent_clean, latent_mean, brain_mask, save_path):
    D, H, W = latent_mean.shape
    fig, ax = plt.subplots(3, 3, figsize=(15, 10))

    views = [
        (latent_mean, "Latent moyen"),
        (brain_mask, "Masque"),
        (latent_clean, "Apres masquage"),
    ]

    slices = [
        lambda v: v[:, :, W // 2],
        lambda v: v[:, H // 2, :],
        lambda v: v[D // 2, :, :],
    ]

    for row, slicer in enumerate(slices):
        for col, (volume, title) in enumerate(views):
            ax[row, col].imshow(slicer(volume), cmap="gray")
            ax[row, col].axis("off")
            if row == 0:
                ax[row, col].set_title(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardee : {save_path}")


# ---------------------------------------------------------------------------
# 3. "Emporte piece" : construction du masque de cerveau
# ---------------------------------------------------------------------------

def compute_brain_mask(magnitude, activation_quantile, occupancy_quantile):
    """
    magnitude : array (N, D, H, W), valeurs >= 0 (deja en valeur absolue,
                eventuellement moyennees sur les canaux). Une valeur haute =
                signal fort.
    Retourne un masque booleen (D, H, W) : True = voxel garde.
    """
    activation_threshold = np.quantile(magnitude.flatten(), activation_quantile)
    print(f"Activation threshold : {activation_threshold:.4f}")

    activation = magnitude > activation_threshold

    # moyenne sur les sujets
    occupancy = activation.astype(float).mean(axis=0)

    occupancy_threshold = np.quantile(occupancy.flatten(), occupancy_quantile)
    print(f"Occupancy threshold : {occupancy_threshold:.4f}")

    candidate = occupancy > occupancy_threshold

    # BFS depuis les 8 coins du volume pour trouver le "fond" connecte
    D, H, W = candidate.shape
    background = np.zeros_like(candidate, dtype=bool)

    corners = [
        (0, 0, 0), (0, 0, W - 1), (0, H - 1, 0), (0, H - 1, W - 1),
        (D - 1, 0, 0), (D - 1, 0, W - 1), (D - 1, H - 1, 0), (D - 1, H - 1, W - 1),
    ]
    neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    queue = deque(corners)
    while queue:
        x, y, z = queue.popleft()
        if not (0 <= x < D and 0 <= y < H and 0 <= z < W):
            continue
        if background[x, y, z] or candidate[x, y, z]:
            continue
        background[x, y, z] = True
        for dx, dy, dz in neighbors:
            queue.append((x + dx, y + dy, z + dz))

    brain_mask = ~background
    print(f"Voxels conserves : {brain_mask.sum()} / Voxels supprimes : {background.sum()}")
    return brain_mask


def apply_mask_and_flatten(X, mask):
    """
    X : (N, C, D, H, W). mask : (D, H, W) booleen ou None.
    Retourne un tableau 2D (N, features) : canaux + voxels selectionnes concatenes.
    """
    N, C = X.shape[0], X.shape[1]
    if mask is None:
        return X.reshape(N, -1)
    return X[:, :, mask].reshape(N, -1)


# ---------------------------------------------------------------------------
# 4. Reduction de dimension
# ---------------------------------------------------------------------------

def build_reduction_model(method, n_components, pca_variance, random_state):
    if method == "pca":
        n = n_components if n_components else pca_variance
        return PCA(n_components=n, random_state=random_state)
    elif method == "truncatedsvd":
        return TruncatedSVD(n_components=n_components or 3, random_state=random_state)
    elif method == "dictionarylearning":
        return DictionaryLearning(
            n_components=n_components or 3,
            transform_algorithm="lasso_lars",
            random_state=random_state,
        )
    elif method == "sparsepca":
        return SparsePCA(n_components=n_components or 3, random_state=random_state)
    else:
        raise ValueError(f"Methode de reduction inconnue : {method}")


def plot_explained_variance(model, pca_variance_target, save_path):
    """Uniquement pour les methodes qui exposent explained_variance_ratio_ (PCA, TruncatedSVD)."""
    if not hasattr(model, "explained_variance_ratio_"):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.cumsum(model.explained_variance_ratio_), label="variance cumulee")
    ax.axhline(pca_variance_target, color="red", linestyle="--")
    ax.set_title("Variance expliquee cumulee")
    ax.set_xlabel("Nombre de composantes")
    ax.set_ylabel("Variance cumulee")
    ax.grid(True)
    ax.legend()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardee : {save_path}")


# ---------------------------------------------------------------------------
# 5. Clustering
# ---------------------------------------------------------------------------

def build_clustering_model(method, n_clusters, random_state, n_init, n_neighbors,
                            gmm_reg_covar, hdbscan_min_cluster_size):
    if method == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    elif method == "spectral":
        return SpectralClustering(
            n_clusters=n_clusters, random_state=random_state,
            affinity="nearest_neighbors", n_neighbors=n_neighbors,
        )
    elif method == "gmm":
        return GaussianMixture(
            n_components=n_clusters, random_state=random_state,
            reg_covar=gmm_reg_covar, covariance_type="full", max_iter=300,
        )
    elif method == "cah":
        return AgglomerativeClustering(n_clusters=n_clusters)
    elif method == "hdbscan":
        return HDBSCAN(min_cluster_size=hdbscan_min_cluster_size)
    else:
        raise ValueError(f"Methode de clustering inconnue : {method}")


def plot_clusters_2d(X_vis, cluster_labels, diagnosis_labels, save_path, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scatter0 = axes[0].scatter(X_vis[:, 0], X_vis[:, 1], c=cluster_labels, cmap="viridis", s=25)
    axes[0].set_title(f"Projection 2D - clusters {title_suffix}")
    axes[0].set_xlabel("Composante 1")
    axes[0].set_ylabel("Composante 2")
    axes[0].grid(True)
    fig.colorbar(scatter0, ax=axes[0], label="Cluster")

    diag_codes, diag_names = pd.factorize(diagnosis_labels)
    scatter1 = axes[1].scatter(X_vis[:, 0], X_vis[:, 1], c=diag_codes, cmap="tab10", s=25)
    axes[1].set_title("Projection 2D - diagnosis")
    axes[1].set_xlabel("Composante 1")
    axes[1].set_ylabel("Composante 2")
    axes[1].grid(True)
    handles = scatter1.legend_elements()[0]
    axes[1].legend(handles, list(diag_names), title="diagnosis", loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardee : {save_path}")


def evaluate_clustering(X, y, labels):
    results = {}
    n_unique = len(set(labels))
    if 1 < n_unique < len(labels):
        results["silhouette"] = silhouette_score(X, labels)
        results["davies_bouldin"] = davies_bouldin_score(X, labels)
        results["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    else:
        results["silhouette"] = None
        results["davies_bouldin"] = None
        results["calinski_harabasz"] = None

    if y is not None:
        results["ARI"] = adjusted_rand_score(y, labels)
        results["NMI"] = normalized_mutual_info_score(y, labels)

    for name, value in results.items():
        print(f"{name} : {value}")
    return results


# ---------------------------------------------------------------------------
# 6. Pipeline principal
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline round-latent + emporte-piece (optionnel) + reduction de dimension + clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_csv", required=True, help="Chemin vers le dataset csv (colonnes: latent_path, diagnosis, subject_id, ...)")
    parser.add_argument("--output_dir", default=None, help="Dossier de sortie (par defaut: outputs/runs/run_<date>)")

    # Canal(aux)
    parser.add_argument("--channel", default="0", help="Numero du canal a utiliser, ou 'fusion' pour combiner tous les canaux")
    parser.add_argument("--round_decimals", type=int, default=3, help="Nombre de decimales pour l'arrondi des latents")

    # Emporte-piece
    parser.add_argument("--mask", dest="use_mask", action=argparse.BooleanOptionalAction, default=True, help="Appliquer le masque de cerveau (emporte-piece)")
    parser.add_argument("--activation_quantile", type=float, default=0.67, help="Quantile pour le seuil d'activation (emporte-piece)")
    parser.add_argument("--occupancy_quantile", type=float, default=0.25, help="Quantile pour le seuil d'occupancy (emporte-piece)")

    # Standardisation
    parser.add_argument("--standardize", dest="standardize", action=argparse.BooleanOptionalAction, default=True, help="Standardiser les features avec StandardScaler")

    # Reduction de dimension
    parser.add_argument("--reduction_method", choices=REDUCTION_METHODS, default="pca", help="Methode de reduction de dimension")
    parser.add_argument("--n_components", type=int, default=None, help="Nombre de composantes pour la reduction (toutes methodes). Pour PCA, si non fourni, --pca_variance est utilise a la place")
    parser.add_argument("--pca_variance", type=float, default=0.70, help="Variance expliquee visee pour PCA (utilise seulement si --n_components n'est pas fourni et methode=pca)")
    parser.add_argument("--vis_components", type=int, default=2, help="Nombre de composantes pour la visualisation 2D (calculee via PCA sur l'espace reduit)")

    # Clustering
    parser.add_argument("--clustering_method", choices=CLUSTERING_METHODS, default="kmeans", help="Methode de clustering")
    parser.add_argument("--n_clusters", type=int, default=2, help="Nombre de clusters (kmeans, spectral, gmm, cah)")
    parser.add_argument("--n_init", type=int, default=10, help="Nombre d'initialisations (kmeans)")
    parser.add_argument("--n_neighbors", type=int, default=8, help="Nombre de voisins (spectral clustering)")
    parser.add_argument("--gmm_reg_covar", type=float, default=1e-3, help="Regularisation de covariance (gmm)")
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=5, help="Taille minimale d'un cluster (hdbscan)")

    parser.add_argument("--random_state", type=int, default=42, help="Graine aleatoire")

    args = parser.parse_args()
    np.random.seed(args.random_state)

    output_dir = args.output_dir or datetime.now().strftime("outputs/runs/analyze/run_%Y%m%d_%H%M%S") #euh c'est pas mieux de garder que l'heure et la minute?run_%Y%m%d_%H%M
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Dossier de sortie : {output_dir}")

    # 1. Chargement + choix du/des canal(aux)
    df, latents = load_latents(args.dataset_csv)
    N = latents.shape[0]
    X_multi, channels = select_channels(latents, args.channel)

    # 2. Arrondi
    X_rounded = np.round(X_multi, decimals=args.round_decimals)
    print(f"Valeurs uniques avant/apres arrondi : {len(np.unique(X_multi))} -> {len(np.unique(X_rounded))}")

    channel_label = "fusion" if len(channels) > 1 else f"canal {channels[0]}"

    plot_latent_alone(
        X_rounded[0, 0], f"Visualisation du latent 1 ({channel_label}, 1er canal affiche)",
        os.path.join(figures_dir, "latent_1.png"),
    )

    latent_mean = X_rounded[:, 0].mean(axis=0)
    plot_latent_alone(
        latent_mean, f"Volume moyen ({channel_label}, 1er canal affiche)",
        os.path.join(figures_dir, "latent_mean.png"),
    )

    # 3. Emporte piece (masque de cerveau) - optionnel
    brain_mask = None
    if args.use_mask:
        magnitude = np.abs(X_rounded).mean(axis=1)  # (N, D, H, W), moyenne des canaux en valeur absolue
        brain_mask = compute_brain_mask(magnitude, args.activation_quantile, args.occupancy_quantile)

        latent_clean = latent_mean.copy()
        latent_clean[~brain_mask] = np.nan
        plot_latent_with_mask(
            latent_clean, latent_mean, brain_mask,
            os.path.join(figures_dir, "latent_with_mask.png"),
        )
    else:
        print("Emporte-piece desactive (--no-mask) : aucun masque applique.")

    X_flat = apply_mask_and_flatten(X_rounded, brain_mask)
    print(f"Shape apres selection canal(aux) + masque : {X_flat.shape}")

    # 4. Standardisation - optionnelle
    if args.standardize:
        X_prepared = StandardScaler().fit_transform(X_flat)
    else:
        X_prepared = X_flat
        print("Standardisation desactivee (--no-standardize).")
    y_encoded = LabelEncoder().fit_transform(df["diagnosis"])

    # 5. Reduction de dimension
    reduction_model = build_reduction_model(
        args.reduction_method, args.n_components, args.pca_variance, args.random_state
    )
    X_reduced = reduction_model.fit_transform(X_prepared)
    print(f"[{args.reduction_method}] shape reduite : {X_reduced.shape}")

    plot_explained_variance(
        reduction_model, args.pca_variance,
        os.path.join(figures_dir, f"{args.reduction_method}_variance.png"),
    )

    # Visualisation 2D (toujours via PCA sur l'espace deja reduit, quelle que soit la methode)
    if X_reduced.shape[1] >= 2:
        X_vis = PCA(n_components=args.vis_components, random_state=args.random_state).fit_transform(X_reduced)
    else:
        X_vis = None
        print("Visualisation 2D ignoree : moins de 2 composantes dans l'espace reduit.")

    # 6. Clustering
    clustering_model = build_clustering_model(
        args.clustering_method, args.n_clusters, args.random_state, args.n_init,
        args.n_neighbors, args.gmm_reg_covar, args.hdbscan_min_cluster_size,
    )
    cluster_labels = clustering_model.fit_predict(X_reduced)
    print(f"[{args.clustering_method}] clusters trouves : {sorted(set(cluster_labels))}")

    if X_vis is not None:
        plot_clusters_2d(
            X_vis, cluster_labels, df["diagnosis"],
            os.path.join(figures_dir, "clusters.png"),
            title_suffix=f"({args.reduction_method} + {args.clustering_method})",
        )
    metrics = evaluate_clustering(X_reduced, y_encoded, cluster_labels)

    # 7. Sauvegardes
    np.save(os.path.join(output_dir, "embeddings.npy"), X_reduced)
    if X_vis is not None:
        np.save(os.path.join(output_dir, "embeddings_2d.npy"), X_vis)

    joblib.dump(clustering_model, os.path.join(output_dir, f"{args.clustering_method}_model.joblib"))
    joblib.dump(reduction_model, os.path.join(output_dir, f"{args.reduction_method}_model.joblib"))
    if brain_mask is not None:
        joblib.dump(brain_mask, os.path.join(output_dir, "brain_mask.joblib"))

    df_out = df.copy()
    df_out["cluster_label"] = cluster_labels
    csv_path = os.path.join(output_dir, "dataset_with_clusters.csv")
    df_out.to_csv(csv_path, index=False)

    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "clustering_metrics.csv"), index=False)

    print("\nTermine.")
    print(f"- Canaux utilises   : {channels} ({channel_label})")
    print(f"- Masque emporte-piece : {'oui' if args.use_mask else 'non'}")
    print(f"- Standardisation   : {'oui' if args.standardize else 'non'}")
    print(f"- Reduction         : {args.reduction_method}")
    print(f"- Clustering        : {args.clustering_method}")
    print(f"- Figures           : {figures_dir}")
    print(f"- Embeddings        : embeddings.npy")
    print(f"- Modeles           : {args.reduction_method}_model.joblib, {args.clustering_method}_model.joblib")
    print(f"- CSV avec labels   : {csv_path}")
    print(f"- Metriques         : clustering_metrics.csv")


if __name__ == "__main__":
    main()
