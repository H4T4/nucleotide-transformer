#!/usr/bin/env python3
"""
Hilfsfunktionen für:
- Cosine-Similarity
- PCA / UMAP
- Plots
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import umap  # Paket: umap-learn


# ---------------------------------------------------------------------------
# Ähnlichkeitsmetriken & Hilfsfunktionen
# ---------------------------------------------------------------------------


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Cosine-Similarity-Matrix für alle Paare in X.

    Parameter
    ---------
    X : np.ndarray
        Shape: (n, d)

    Rückgabewert
    ------------
    np.ndarray
        Shape: (n, n) mit Cosine-Similarity.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / (norms + 1e-12)
    return X_norm @ X_norm.T


def cosine_to_reference(X: np.ndarray, ref_idx: int) -> np.ndarray:
    """
    Cosine-Similarity jedes Vektors in X zu einem Referenzvektor X[ref_idx].

    Parameter
    ---------
    X : np.ndarray
        Shape: (n, d)
    ref_idx : int
        Index des Referenzvektors.

    Rückgabewert
    ------------
    np.ndarray
        Shape: (n,), Cosine-Similarity zwischen X[i] und X[ref_idx].
    """
    ref = X[ref_idx]
    ref_norm = np.linalg.norm(ref) + 1e-12
    dots = X @ ref
    norms = np.linalg.norm(X, axis=1) * ref_norm + 1e-12
    return dots / norms


def euclidean_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Euklidische Distanz-Matrix für alle Paare in X.

    Parameter
    ---------
    X : np.ndarray
        Shape: (n, d)

    Rückgabewert
    ------------
    np.ndarray
        Shape: (n, n) mit euklidischen Distanzen.
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    norms_sq = np.sum(X**2, axis=1, keepdims=True)  # (n, 1)
    dists_sq = norms_sq + norms_sq.T - 2.0 * (X @ X.T)  # (n, n)
    dists_sq = np.maximum(dists_sq, 0.0)  # numerisch stabil
    return np.sqrt(dists_sq)


def euclidean_to_reference(X: np.ndarray, ref_idx: int) -> np.ndarray:
    """
    Euklidische Distanz jedes Vektors in X zu einem Referenzvektor X[ref_idx].

    Parameter
    ---------
    X : np.ndarray
        Shape: (n, d)
    ref_idx : int
        Index des Referenzvektors.

    Rückgabewert
    ------------
    np.ndarray
        Shape: (n,), euklidische Distanzen zwischen X[i] und X[ref_idx].
    """
    ref = X[ref_idx]  # (d,)
    diffs = X - ref  # (n, d)
    return np.linalg.norm(diffs, axis=1)  # (n,)


def summarize_matrix(name: str, M: np.ndarray) -> None:
    """
    Gibt einige Kennzahlen für eine Matrix aus (min, max, Mittelwert),
    Diagonale wird ignoriert.
    """
    n = M.shape[0]
    if n <= 1:
        print(f"{name}: Matrix zu klein für sinnvolle Statistik (n={n}).")
        return

    mask = ~np.eye(n, dtype=bool)
    vals = M[mask]

    print(f"{name}:")
    print(f"  Shape: {M.shape}")
    print(f"  Min:   {vals.min():.4f}")
    print(f"  Max:   {vals.max():.4f}")
    print(f"  Mean:  {vals.mean():.4f}")


# ---------------------------------------------------------------------------
# PCA & UMAP
# ---------------------------------------------------------------------------


def pca_2d(X: np.ndarray) -> np.ndarray:
    """
    Einfache PCA auf 2D mit SVD.

    Parameter
    ---------
    X : np.ndarray
        Datenmatrix (n_samples, n_features)

    Rückgabewert
    ------------
    np.ndarray
        PCA-Projektion auf 2D, Shape: (n_samples, 2)
    """
    X_centered = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:2]  # erste 2 Hauptkomponenten
    X_pca = X_centered @ components.T
    return X_pca


def umap_2d(
    X: np.ndarray,
    n_neighbors: int = 5,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 0,
) -> np.ndarray:
    """
    UMAP-Projektion auf 2D.

    Parameter
    ---------
    X : np.ndarray
        Datenmatrix (n_samples, n_features)

    Rückgabewert
    ------------
    np.ndarray
        UMAP-Projektion auf 2D, Shape: (n_samples, 2)
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(X)


# ---------------------------------------------------------------------------
# Plot-Funktionen
# ---------------------------------------------------------------------------


def plot_pca(X_pca: np.ndarray, lengths: np.ndarray, output_path: Path) -> None:
    """
    PCA-Scatterplot, farbkodiert nach Sequenzlänge.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    sc = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        s=60,
        c=lengths,
    )

    for i, l in enumerate(lengths):
        ax.text(
            X_pca[i, 0],
            X_pca[i, 1],
            f"{int(l)}",
            fontsize=8,
            ha="center",
            va="center",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA der Sequenz-Embeddings (farbkodiert nach Sequenzlänge)")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Sequenzlänge (bp)")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"PCA-Plot gespeichert unter: {output_path}")


def plot_umap(X_umap: np.ndarray, lengths: np.ndarray, output_path: Path) -> None:
    """
    UMAP-Scatterplot, farbkodiert nach Sequenzlänge.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    sc = ax.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        s=60,
        c=lengths,
    )

    for i, l in enumerate(lengths):
        ax.text(
            X_umap[i, 0],
            X_umap[i, 1],
            f"{int(l)}",
            fontsize=8,
            ha="center",
            va="center",
        )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("UMAP der Sequenz-Embeddings (farbkodiert nach Sequenzlänge)")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Sequenzlänge (bp)")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"UMAP-Plot gespeichert unter: {output_path}")


def plot_cosine_vs_length(
    lengths: np.ndarray, cos_to_ref: np.ndarray, ref_length: int, output_path: Path
) -> None:
    """
    Cosine-Similarity zur Referenz-Sequenz als Funktion der Sequenzlänge.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(lengths, cos_to_ref, marker="o")
    ax.set_xlabel("Sequenzlänge (bp)")
    ax.set_ylabel(f"Cosine-Similarity zur Referenz (Länge {ref_length} bp)")
    ax.set_title("Ähnlichkeit der Sequenz-Embeddings vs. Kontextlänge")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Cosine-vs-Länge-Plot gespeichert unter: {output_path}")


def plot_distance_vs_length(
    lengths: np.ndarray, dist_to_ref: np.ndarray, ref_length: int, output_path: Path
) -> None:
    """
    Plottet die euklidische Distanz zur Referenzsequenz (z.B. 60 bp)
    als Funktion der Sequenzlänge.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(lengths, dist_to_ref, marker="o")
    ax.set_xlabel("Sequenzlänge (bp)")
    ax.set_ylabel(f"Euklidische Distanz zur Referenz (Länge {ref_length} bp)")
    ax.set_title("Distanz der Sequenz-Embeddings vs. Kontextlänge")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Distance-vs-Länge-Plot gespeichert unter: {output_path}")
