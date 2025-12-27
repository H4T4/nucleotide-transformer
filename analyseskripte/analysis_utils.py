#!/usr/bin/env python3
"""
Hilfsfunktionen für:
- Cosine-Similarity & euklidische Distanz
- SNP-spezifische Statistiken
- PCA / UMAP
- Plots
"""

from pathlib import Path
from typing import Dict, List

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

    (Aktuell in run_agront_analysis.py nicht verwendet, bleibt aber für
    spätere Experimente erhalten.)
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

    (Aktuell nicht verwendet, bleibt der Vollständigkeit halber erhalten.)
    """
    ref = X[ref_idx]  # (d,)
    diffs = X - ref  # (n, d)
    return np.linalg.norm(diffs, axis=1)  # (n,)


def pairwise_cosine_and_distance(
    emb_group: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Berechnet alle paarweisen Cosine-Similarities und euklidischen Distanzen
    innerhalb einer kleinen Embedding-Gruppe (z.B. 2–4 Sequenzen).

    emb_group : np.ndarray, Shape (k, d)

    Rückgabe:
      cos_vals  : 1D-Array aller Cosine-Werte, Länge k*(k-1)/2
      dist_vals : 1D-Array aller Distanz-Werte, Länge k*(k-1)/2
    """
    k = emb_group.shape[0]
    if k < 2:
        return np.array([]), np.array([])

    cos_mat = cosine_similarity_matrix(emb_group)  # (k, k)
    dist_mat = euclidean_distance_matrix(emb_group)  # (k, k)

    # obere Dreiecksmatrix ohne Diagonale (i < j)
    mask = np.triu(np.ones((k, k), dtype=bool), k=1)
    cos_vals = cos_mat[mask]
    dist_vals = dist_mat[mask]
    return cos_vals, dist_vals


def compute_snp_stats_by_length(
    sequences: List[str],
    lengths: np.ndarray,
    embeddings: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """
    Berechnet pro Sequenzlänge Statistiken zwischen Sequenzen, die sich nur in der
    Base in der Mitte unterscheiden (gleiche Flanken = gleiche Template-Sequenz).

    Für jede Länge L werden:
      - globale Mittelwerte über ALLE SNP-Paare (mean_cosine, mean_distance)
      - Anzahl aller SNP-Paare (n_pairs)
      - Mittelwerte pro Template-Sequenz (template_mean_cosines/-distances)
        gespeichert.

    Rückgabe:
      dict:
        length -> {
            "mean_cosine": float,
            "mean_distance": float,
            "n_pairs": int,
            "template_mean_cosines": List[float],
            "template_mean_distances": List[float],
        }
    """
    results: Dict[int, Dict[str, float]] = {}

    unique_lengths = np.unique(lengths)
    for L in unique_lengths:
        idx_L = np.where(lengths == L)[0]
        if len(idx_L) == 0:
            continue

        mid = L // 2
        # Gruppieren nach Flanken (ohne Mittel-Base)
        groups: Dict[str, List[int]] = {}
        for i in idx_L:
            seq = sequences[i]
            flanks = seq[:mid] + seq[mid + 1 :]  # alles außer SNP-Position
            groups.setdefault(flanks, []).append(i)

        all_cos_vals = []
        all_dist_vals = []
        template_mean_cosines: List[float] = []
        template_mean_distances: List[float] = []

        for flanks, idxs in groups.items():
            if len(idxs) < 2:
                continue
            emb_group = embeddings[idxs, :]  # (k, d) mit k=2..4
            cos_vals, dist_vals = pairwise_cosine_and_distance(emb_group)
            if cos_vals.size == 0:
                continue

            # globale Sammlung aller paarweisen Werte
            all_cos_vals.append(cos_vals)
            all_dist_vals.append(dist_vals)

            # Mittelwerte pro Template (für den Plot pro Templatesequenz)
            template_mean_cosines.append(float(cos_vals.mean()))
            template_mean_distances.append(float(dist_vals.mean()))

        if not all_cos_vals:
            continue

        cos_concat = np.concatenate(all_cos_vals)
        dist_concat = np.concatenate(all_dist_vals)

        results[int(L)] = {
            "mean_cosine": float(cos_concat.mean()),
            "mean_distance": float(dist_concat.mean()),
            "n_pairs": int(cos_concat.size),
            "template_mean_cosines": template_mean_cosines,
            "template_mean_distances": template_mean_distances,
        }

    return results


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
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
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

    (Aktuell nicht in run_agront_analysis.py verwendet, bleibt aber verfügbar.)
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

    (Aktuell nicht in run_agront_analysis.py verwendet, bleibt aber verfügbar.)
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


def plot_snp_stats_by_length(
    snp_stats: Dict[int, Dict[str, float]],
    output_cosine_path: Path,
    output_distance_path: Path,
) -> None:
    """
    Plottet die mittlere Cosine-Similarity bzw. euklidische Distanz
    zwischen SNP-Varianten (gleiche Flanken) als Funktion der Sequenzlänge.

    Es wird NICHT nur ein Wert pro Länge geplottet, sondern
    für jede Templatesequenz ein Punkt:
      x = Sequenzlänge L
      y = Mittelwert der paarweisen Werte innerhalb der Varianten
          dieses Templates.
    """
    if not snp_stats:
        print("Keine SNP-Statistiken vorhanden – überspringe SNP-Plots.")
        return

    lengths_per_template: List[int] = []
    cos_per_template: List[float] = []
    dist_per_template: List[float] = []

    for L in sorted(snp_stats.keys()):
        stats_L = snp_stats[L]
        tm_cos = stats_L.get("template_mean_cosines", [])
        tm_dist = stats_L.get("template_mean_distances", [])
        if len(tm_cos) != len(tm_dist):
            continue
        for c_val, d_val in zip(tm_cos, tm_dist):
            lengths_per_template.append(L)
            cos_per_template.append(c_val)
            dist_per_template.append(d_val)

    if not lengths_per_template:
        print("Keine Template-basierten SNP-Statistiken gefunden – keine SNP-Plots.")
        return

    lengths_arr = np.asarray(lengths_per_template, dtype=int)
    cos_arr = np.asarray(cos_per_template, dtype=float)
    dist_arr = np.asarray(dist_per_template, dtype=float)

    x_min = lengths_arr.min()
    x_max = lengths_arr.max()

    # Plot: mittlere Cosine-Similarity pro Template
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(lengths_arr, cos_arr, s=20)
    ax.set_xlabel("Sequenzlänge (bp)")
    ax.set_ylabel(
        "mittlere Cosine-Similarity\n(zwischen SNP-Varianten eines Templates)"
    )
    ax.set_title("SNP-Embeddings: mittlere Cosine-Similarity vs. Kontextlänge")
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_cosine_path)
    plt.close(fig)
    print(f"SNP-Cosine-Plot gespeichert unter: {output_cosine_path}")

    # Plot: mittlere euklidische Distanz pro Template
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(lengths_arr, dist_arr, s=20)
    ax.set_xlabel("Sequenzlänge (bp)")
    ax.set_ylabel(
        "mittlere euklidische Distanz\n(zwischen SNP-Varianten eines Templates)"
    )
    ax.set_title("SNP-Embeddings: mittlere Distanz vs. Kontextlänge")
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_distance_path)
    plt.close(fig)
    print(f"SNP-Distanz-Plot gespeichert unter: {output_distance_path}")


# ---------------------------------------------------------------------------
# Distanz-Statistik über alle Sequenzen einer Länge
# ---------------------------------------------------------------------------


def mean_distance_by_length(
    lengths: np.ndarray,
    embeddings: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """
    Berechnet pro Sequenzlänge die mittlere euklidische Distanz zwischen
    ALLEN Sequenzen dieser Länge.

    Die Funktion arbeitet analog zur SNP-Auswertung, verzichtet aber auf
    die Gruppierung nach Templates. So entsteht genau ein Mittelwert pro
    Länge, der als Referenz für zufällige Sequenzsets genutzt werden kann.

    Rückgabe:
      dict:
        length -> {"mean_distance": float, "n_pairs": int}
    """
    results: Dict[int, Dict[str, float]] = {}

    unique_lengths = np.unique(lengths)
    for L in unique_lengths:
        idx_L = np.where(lengths == L)[0]
        if len(idx_L) < 2:
            # mit nur einer Sequenz ist keine Distanz definierbar
            continue

        emb_L = embeddings[idx_L, :]  # (n_L, d)
        dist_mat = euclidean_distance_matrix(emb_L)

        # obere Dreiecksmatrix ohne Diagonale, um alle Paar-Kombinationen zu mitteln
        mask = np.triu(np.ones(dist_mat.shape, dtype=bool), k=1)
        dist_vals = dist_mat[mask]

        results[int(L)] = {
            "mean_distance": float(dist_vals.mean()),
            "n_pairs": int(dist_vals.size),
        }

    return results


def plot_mean_distance_by_length(
    distance_stats: Dict[int, Dict[str, float]],
    output_path: Path,
) -> None:
    """
    Plottet genau einen Punkt pro Sequenzlänge: die mittlere euklidische Distanz
    zwischen allen Sequenzen dieser Länge (z.B. 1000 Zufallssequenzen).
    """
    if not distance_stats:
        print("Keine Distanz-Statistiken pro Länge vorhanden – überspringe Plot.")
        return

    lengths_sorted = sorted(distance_stats.keys())
    mean_dists = [distance_stats[L]["mean_distance"] for L in lengths_sorted]

    x_min = lengths_sorted[0]
    x_max = lengths_sorted[-1]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Ein Punkt pro Länge; eine dünne Linie hilft beim Verlauf über die Länge
    ax.plot(lengths_sorted, mean_dists, linestyle="--", color="#1f77b4", alpha=0.6)
    ax.scatter(lengths_sorted, mean_dists, s=40, color="#1f77b4")

    ax.set_xlabel("Sequenzlänge (bp)")
    ax.set_ylabel("mittlere euklidische Distanz\n(alle Sequenzpaare pro Länge)")
    ax.set_title("Seq-Embeddings: mittlere Distanz vs. Länge")
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Plot 'mittlere Distanz vs. Länge' gespeichert unter: {output_path}")


# ---------------------------------------------------------------------------
# SNP-Distanz relativ zur globalen Durchschnittsdistanz (in Prozent)
# ---------------------------------------------------------------------------


def compute_snp_distance_percent_by_length(
    snp_stats: Dict[int, Dict[str, float]],
    length_distance_stats: Dict[int, Dict[str, float]],
) -> Dict[int, Dict[str, float]]:
    """
    Setzt die mittlere SNP-Distanz pro Länge in Relation zur globalen
    Durchschnittsdistanz der jeweiligen Länge.

    Ergebnis:
      percent = (SNP-mean-distance / mean-distance-alle-Paare) * 100

    Rückgabe:
      dict:
        length -> {
            "percent": float,
            "snp_mean_distance": float,
            "baseline_distance": float,
        }
    """
    results: Dict[int, Dict[str, float]] = {}

    for L, stats_L in snp_stats.items():
        if L not in length_distance_stats:
            # Falls für eine Länge keine globale Distanz existiert, überspringen.
            continue

        baseline = length_distance_stats[L]["mean_distance"]
        if baseline <= 0:
            # Numerisch oder inhaltlich nicht sinnvoll, dann keine Prozentangabe.
            continue

        snp_mean = stats_L["mean_distance"]
        percent = (snp_mean / baseline) * 100.0

        results[int(L)] = {
            "percent": float(percent),
            "snp_mean_distance": float(snp_mean),
            "baseline_distance": float(baseline),
        }

    return results


def plot_snp_distance_percent_by_length(
    distance_percent_stats: Dict[int, Dict[str, float]],
    output_path: Path,
) -> None:
    """
    Plottet die mittlere SNP-Distanz pro Länge als Prozentwert der
    globalen Durchschnittsdistanz (alle Sequenzpaare pro Länge).
    """
    if not distance_percent_stats:
        print("Keine Prozent-Statistiken für SNP-Distanzen vorhanden – überspringe Plot.")
        return

    lengths_sorted = sorted(distance_percent_stats.keys())
    percents = [distance_percent_stats[L]["percent"] for L in lengths_sorted]

    x_min = lengths_sorted[0]
    x_max = lengths_sorted[-1]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Ein Punkt pro Länge mit kurzer Trendlinie für die Lesbarkeit.
    ax.plot(lengths_sorted, percents, linestyle="--", color="#d62728", alpha=0.6)
    ax.scatter(lengths_sorted, percents, s=40, color="#d62728")

    ax.set_xlabel("Sequenzlänge (bp)")
    ax.set_ylabel("SNP-Distanz relativ zur mittleren Distanz [%]")
    ax.set_title("SNP-Embeddings: Distanz in % der mittleren Länge-Distanz")
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Plot 'SNP-Distanz in %' gespeichert unter: {output_path}")


# ---------------------------------------------------------------------------
# Mittlere Embedding-Norm pro Länge
# ---------------------------------------------------------------------------


def mean_embedding_norm_by_length(
    lengths: np.ndarray,
    embeddings: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """
    Berechnet die mittlere L2-Norm der Embeddings pro Sequenzlänge.

    Rückgabe:
      dict:
        length -> {"mean_norm": float, "n_sequences": int}
    """
    results: Dict[int, Dict[str, float]] = {}

    norms = np.linalg.norm(embeddings, axis=1)
    unique_lengths = np.unique(lengths)

    for L in unique_lengths:
        idx_L = np.where(lengths == L)[0]
        if len(idx_L) == 0:
            continue

        mean_norm = float(norms[idx_L].mean())
        results[int(L)] = {
            "mean_norm": mean_norm,
            "n_sequences": int(len(idx_L)),
        }

    return results


def plot_mean_embedding_norm_by_length(
    norm_stats: Dict[int, Dict[str, float]],
    output_path: Path,
) -> None:
    """
    Plottet pro Sequenzlänge die mittlere L2-Norm der Embeddings.
    """
    if not norm_stats:
        print("Keine Norm-Statistiken pro Länge vorhanden – überspringe Plot.")
        return

    lengths_sorted = sorted(norm_stats.keys())
    mean_norms = [norm_stats[L]["mean_norm"] for L in lengths_sorted]

    x_min = lengths_sorted[0]
    x_max = lengths_sorted[-1]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Ein Punkt pro Länge, ergänzt um eine dünne Linie zur Orientierung.
    ax.plot(lengths_sorted, mean_norms, linestyle="--", color="#2ca02c", alpha=0.6)
    ax.scatter(lengths_sorted, mean_norms, s=40, color="#2ca02c")

    ax.set_xlabel("Sequenzlänge (bp)")
    ax.set_ylabel("mittlere Embedding-Norm (L2)")
    ax.set_title("Seq-Embeddings: mittlere Norm vs. Länge")
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Plot 'mittlere Embedding-Norm vs. Länge' gespeichert unter: {output_path}")
