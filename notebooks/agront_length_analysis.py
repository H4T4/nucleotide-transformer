#!/usr/bin/env python3
"""
Analyse-Skript für AgroNT-Embeddings unterschiedlicher Kontextlängen.

Funktionen des Skripts
----------------------
1. Liest alle FASTA-Dateien in einem Verzeichnis ein.
2. Lädt ein AgroNT-Modell (z.B. 1B_agro_nt) mit passender max_positions.
3. Berechnet Embeddings für alle Sequenzen:
   - Fokus: Embedding an der mittleren Position (SNP-Position).
4. Führt folgende Analysen durch:
   - Cosine-Similarity-Matrix über alle Middle-Position-Embeddings.
   - PCA (2D) auf den Middle-Position-Embeddings.
   - UMAP (2D) auf den Middle-Position-Embeddings.
   - Cosine-Similarity aller Sequenzen zur 60-bp-Sequenz
     (bzw. zur längsten Sequenz, falls nicht exakt 60 bp existiert).
5. Speichert Plots:
   - pca_middle_embeddings.png
   - umap_middle_embeddings.png
   - cosine_vs_length.png

Verwendung:
python agront_length_analysis.py \
  --fasta-dir /home/htamm/models/agront/nucleotide-transformer/notebooks/fasta_files \
  --model-name 1B_agro_nt \
  --layer 12 \
  --output-dir ./agront_plots

"""

import torch
import os

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import umap  # Paket: umap-learn

from nucleotide_transformer.pretrained import get_pretrained_model


def erste_gpu():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print(torch.cuda.device_count())


# ---------------------------------------------------------------------------
# FASTA-Handling
# ---------------------------------------------------------------------------


def read_fasta_sequences(path: Path) -> List[str]:
    """
    Sehr einfacher FASTA-Parser ohne zusätzliche Bibliotheken.

    Parameter
    ---------
    path : Path
        Pfad zur FASTA-Datei.

    Rückgabewert
    ------------
    List[str]
        Liste der gefundenen Sequenzen (nur ACGTN, Großbuchstaben).
    """
    sequences: List[str] = []
    current: List[str] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                # Leere Zeilen ignorieren
                continue
            if line.startswith(">"):
                # Header-Zeile: ggf. aktuelle Sequenz abschließen
                if current:
                    sequences.append("".join(current).upper())
                    current = []
            else:
                # Zeile gehört zur Sequenz
                current.append(line)
        # letzte Sequenz nicht vergessen
        if current:
            sequences.append("".join(current).upper())

    return sequences


def collect_all_sequences(fasta_dir: Path) -> Dict[str, List[str]]:
    """
    Liest alle FASTA-Dateien in einem Verzeichnis und gibt ein Dict zurück.

    Parameter
    ---------
    fasta_dir : Path
        Verzeichnis, in dem nach FASTA-Dateien gesucht wird (*.fa*).

    Rückgabewert
    ------------
    Dict[str, List[str]]
        Keys: Dateiname (ohne Pfad)
        Values: Liste der Sequenzen in dieser Datei.
    """
    fasta_files = sorted(list(fasta_dir.glob("*.fa*")))
    if not fasta_files:
        raise FileNotFoundError(f"Keine FASTA-Dateien in {fasta_dir} gefunden.")

    all_data: Dict[str, List[str]] = {}
    for fasta_file in fasta_files:
        seqs = read_fasta_sequences(fasta_file)
        if not seqs:
            print(f"Warnung: Keine Sequenzen in {fasta_file} gefunden.")
            continue
        all_data[fasta_file.name] = seqs

    if not all_data:
        raise RuntimeError(
            "Es wurden zwar FASTA-Dateien gefunden, aber keine Sequenzen."
        )
    return all_data


# ---------------------------------------------------------------------------
# Modell & Embeddings
# ---------------------------------------------------------------------------


def load_agront_model(
    max_len: int, model_name: str = "1B_agro_nt", layer_to_save: int = 12
):
    """
    Lädt das AgroNT-Modell mit gewünschter maximaler Position und Layer-Ausgabe.

    Parameter
    ---------
    max_len : int
        Maximale Sequenzlänge aus allen FASTA-Dateien.
    model_name : str
        Name des Modells, z.B. '1B_agro_nt'.
    layer_to_save : int
        Layer-ID, deren Embeddings gespeichert werden (z.B. 12).

    Rückgabewert
    ------------
    parameters, transformed_forward_fn, tokenizer, config
    """
    print(f"Lade Modell '{model_name}' mit max_positions={max_len} ...")
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        embeddings_layers_to_save=(layer_to_save,),
        max_positions=max_len,
    )
    # Haiku benötigt Transformierung der Forward-Funktion,
    # damit wir .apply() aufrufen können.
    forward_fn = hk.transform(forward_fn)
    return parameters, forward_fn, tokenizer, config


def get_last_layer_embeddings(
    sequences: List[str],
    parameters,
    forward_fn,
    tokenizer,
    rng_key,
    layer_to_save: int,
) -> np.ndarray:
    """
    Berechnet die Embeddings eines bestimmten Layers für eine Liste von Sequenzen.

    Parameter
    ---------
    sequences : List[str]
        Nucleotidsequenzen.
    parameters, forward_fn, tokenizer, rng_key :
        Objekte aus load_agront_model.
    layer_to_save : int
        Der Layer, der in embeddings_layers_to_save vorgesehen ist.

    Rückgabewert
    ------------
    np.ndarray
        Embeddings mit Shape (batch, seq_len, hidden_dim)
    """
    # Tokenisierung gemäß Interface von nucleotide_transformer:
    # tokenizer.batch_tokenize(sequences)
    # liefert für jede Sequenz z.B. (token_string, token_ids, attention_mask, ...)
    token_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
    tokens = jnp.asarray(token_ids, dtype=jnp.int32)  # (batch, seq_len)

    outs = forward_fn.apply(parameters, rng_key, tokens)
    key = f"embeddings_{layer_to_save}"
    if key not in outs:
        raise KeyError(
            f"{key} nicht im Model-Output gefunden. Verfügbare Keys: {list(outs.keys())}"
        )

    emb = outs[key]  # (batch, seq_len, hidden_dim)
    return np.array(emb)


# ---------------------------------------------------------------------------
# Embedding-Reduktion (Pooling)
# ---------------------------------------------------------------------------


def get_middle_position_embedding(emb_last_layer: np.ndarray) -> np.ndarray:
    """
    Embedding an der mittleren Position der Sequenz extrahieren.

    Annahme: Der SNP liegt in der Mitte der Sequenz.

    Parameter
    ---------
    emb_last_layer : np.ndarray
        Shape: (batch, seq_len, hidden_dim)

    Rückgabewert
    ------------
    np.ndarray
        Shape: (batch, hidden_dim) – nur das Embedding der mittleren Position.
    """
    batch, seq_len, hidden_dim = emb_last_layer.shape
    mid_idx = seq_len // 2  # bei gerader Länge eine der beiden mittleren Positionen
    return emb_last_layer[:, mid_idx, :]


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


def summarize_matrix(name: str, M: np.ndarray):
    """
    Gibt einige Kennzahlen für eine Matrix aus (min, max, Mittelwert).

    Die Diagonale wird ignoriert, da dort oft triviale Werte stehen (z.B. 1).

    Parameter
    ---------
    name : str
        Name der Matrix (für Print).
    M : np.ndarray
        Matrix (z.B. Similarity- oder Distanzmatrix).
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
    n_neighbors : int
        Typische lokale Nachbarschaftsgröße (UMAP-Parameter).
    min_dist : float
        Minimaler Abstand der Punkte in der Projektion.
    metric : str
        Distanzmetrik, z.B. 'euclidean' oder 'cosine'.
    random_state : int
        Seed für Reproduzierbarkeit.

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


def plot_pca(X_pca: np.ndarray, lengths: np.ndarray, output_path: Path):
    """
    Erstellt einen PCA-Scatterplot, farbkodiert nach Sequenzlänge,
    und speichert ihn als PNG.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    sc = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        s=60,
        c=lengths,
    )

    # Punkte zusätzlich mit der Länge labeln (optional, hier sinnvoll bei wenigen Punkten)
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
    ax.set_title("PCA der Middle-Position-Embeddings (farbkodiert nach Sequenzlänge)")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Sequenzlänge (bp)")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"PCA-Plot gespeichert unter: {output_path}")


def plot_umap(X_umap: np.ndarray, lengths: np.ndarray, output_path: Path):
    """
    Erstellt einen UMAP-Scatterplot, farbkodiert nach Sequenzlänge,
    und speichert ihn als PNG.
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
    ax.set_title("UMAP der Middle-Position-Embeddings (farbkodiert nach Sequenzlänge)")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Sequenzlänge (bp)")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"UMAP-Plot gespeichert unter: {output_path}")


def plot_cosine_vs_length(
    lengths: np.ndarray, cos_to_ref: np.ndarray, ref_length: int, output_path: Path
):
    """
    Plottet die Cosine-Similarity zur Referenzsequenz (z.B. 60 bp)
    als Funktion der Sequenzlänge.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(lengths, cos_to_ref, marker="o")
    ax.set_xlabel("Sequenzlänge (bp)")
    ax.set_ylabel(f"Cosine-Similarity zur Referenz (Länge {ref_length} bp)")
    ax.set_title("Ähnlichkeit der Middle-Position-Embeddings vs. Kontextlänge")

    # Leichte Gitterlinien zur Orientierung
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Cosine-vs-Länge-Plot gespeichert unter: {output_path}")


# ---------------------------------------------------------------------------
# Hauptlogik
# ---------------------------------------------------------------------------


def main():
    erste_gpu()

    parser = argparse.ArgumentParser(
        description="Analyse von AgroNT Middle-Position-Embeddings für Sequenzen aus FASTA-Dateien "
        "mit PCA, UMAP und Cosine-Similarity zur 60-bp-Sequenz."
    )
    parser.add_argument(
        "--fasta-dir",
        type=str,
        required=True,
        help="Verzeichnis mit FASTA-Dateien (*.fa*).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="1B_agro_nt",
        help="AgroNT-Modellname (Default: 1B_agro_nt).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Layer-ID für Embeddings (Default: 12).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="agront_outputs",
        help="Ausgabeverzeichnis für Plots (wird bei Bedarf angelegt).",
    )
    args = parser.parse_args()

    fasta_dir = Path(args.fasta_dir)
    if not fasta_dir.is_dir():
        raise NotADirectoryError(f"{fasta_dir} ist kein Verzeichnis.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) FASTA einlesen
    print(f"Suche FASTA-Dateien in: {fasta_dir}")
    all_sequences_per_file = collect_all_sequences(fasta_dir)

    # Alle Sequenzen und zugehörige Labels (Dateiname, Länge) einsammeln
    all_labels: List[Tuple[str, int]] = []
    all_seqs_global: List[str] = []

    for fname, seqs in all_sequences_per_file.items():
        for s in seqs:
            all_labels.append((fname, len(s)))
            all_seqs_global.append(s)

    lengths = np.array([l for (_, l) in all_labels], dtype=int)

    print("Reihenfolge der Sequenzen:")
    for idx, (fname, length) in enumerate(all_labels):
        print(f"  Index {idx}: Datei={fname}, Länge={length}")

    # 2) Modell laden
    max_len = int(lengths.max())
    print(f"\nMaximale Sequenzlänge über alle Dateien: {max_len}")

    parameters, forward_fn, tokenizer, config = load_agront_model(
        max_len=max_len,
        model_name=args.model_name,
        layer_to_save=args.layer,
    )

    # Fixer Random Key (deterministische Inferenz)
    rng = jax.random.PRNGKey(0)

    # 3) Embeddings für alle Sequenzen berechnen
    emb_all = get_last_layer_embeddings(
        sequences=all_seqs_global,
        parameters=parameters,
        forward_fn=forward_fn,
        tokenizer=tokenizer,
        rng_key=rng,
        layer_to_save=args.layer,
    )  # (N, seq_len, hidden_dim)

    N, seq_len_all, hidden_dim_all = emb_all.shape
    print(
        f"\nGlobale Embeddings-Shape: (N={N}, seq_len={seq_len_all}, hidden_dim={hidden_dim_all})"
    )

    # 4) Middle-Position-Embeddings extrahieren
    middle_all = get_middle_position_embedding(emb_all)  # (N, hidden_dim)

    # 5) Cosine-Similarity-Matrix und kurze Statistik
    cos_middle = cosine_similarity_matrix(middle_all)
    summarize_matrix("GLOBAL Middle-Position Cosine-Similarity", cos_middle)

    # 6) Referenzsequenz bestimmen: bevorzugt Länge 60, sonst längste Sequenz
    target_length = 60
    candidate_indices = np.where(lengths == target_length)[0]
    if len(candidate_indices) > 0:
        ref_idx = int(candidate_indices[0])
    else:
        # falls keine Sequenz mit exakt 60 bp existiert, nimm die längste
        ref_idx = int(np.argmax(lengths))
        target_length = int(lengths[ref_idx])
        print(
            f"\nKeine Sequenz mit exakt 60 bp gefunden – "
            f"verwende Länge {target_length} bp als Referenz."
        )

    # Cosine zu Referenz berechnen
    cos_to_ref = cosine_to_reference(middle_all, ref_idx=ref_idx)

    # 7) PCA & UMAP berechnen
    X_pca = pca_2d(middle_all)
    X_umap = umap_2d(
        middle_all, n_neighbors=5, min_dist=0.1, metric="cosine", random_state=0
    )

    # 8) Plots erstellen und speichern
    plot_pca(
        X_pca=X_pca,
        lengths=lengths,
        output_path=output_dir / "pca_middle_embeddings.png",
    )

    plot_umap(
        X_umap=X_umap,
        lengths=lengths,
        output_path=output_dir / "umap_middle_embeddings.png",
    )

    plot_cosine_vs_length(
        lengths=lengths,
        cos_to_ref=cos_to_ref,
        ref_length=target_length,
        output_path=output_dir / "cosine_vs_length.png",
    )

    print("\nAnalyse abgeschlossen.")


if __name__ == "__main__":
    main()
