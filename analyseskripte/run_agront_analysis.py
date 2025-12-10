#!/usr/bin/env python3
"""
run_agront_analysis.py

Pipeline für:
- Sequenzgenerierung (synthetisch) ODER FASTA-Import
- Embedding-Berechnung mit AgroNT
- globale Cosine-/Distanz-Statistiken
- SNP-spezifische mittlere Cosine/Distanz vs. Kontextlänge
- PCA- und UMAP-Plots

Verwendung (synthetische Sequenzen):

python run_agront_analysis.py \
  --mode synthetic \
  --min-len 6 \
  --max-len 120 \
  --step 6 \
  --num-per-length 1000 \
  --model-name 1B_agro_nt \
  --layer 12 \
  --max-positions 126 \
  --pooling seq \
  --batch-size 512 \
  --output-dir agront_outputs_synthetic

Verwendung (FASTA):

python run_agront_analysis.py \
  --mode fasta \
  --fasta-dir /pfad/zu/fasta_files \
  --model-name 1B_agro_nt \
  --layer 12 \
  --max-positions 126 \
  --pooling seq \
  --batch-size 512 \
  --output-dir agront_outputs_fasta
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import jax

from sequence_utils import generate_snp_variant_sequences, collect_all_sequences
from agront_model import (
    select_first_gpu,
    load_agront_model,
    get_sequence_embeddings,
)
from analysis_utils import (
    cosine_similarity_matrix,
    euclidean_distance_matrix,
    summarize_matrix,
    pca_2d,
    umap_2d,
    plot_pca,
    plot_umap,
    compute_snp_stats_by_length,
    plot_snp_stats_by_length,
)


# ---------------------------------------------------------------------------
# Hilfsfunktionen für Sequenz-Setup
# ---------------------------------------------------------------------------


def build_sequences_from_mode(args) -> Tuple[List[str], np.ndarray]:
    """
    Erzeugt oder lädt Sequenzen abhängig vom Modus.

    Rückgabe:
      sequences : List[str]
      lengths   : np.ndarray[int] mit gleicher Länge wie sequences
    """
    if args.mode == "synthetic":
        sequences, lengths = generate_snp_variant_sequences(
            min_len=args.min_len,
            max_len=args.max_len,
            step=args.step,
            per_length=args.num_per_length,
            seed=args.seed,
        )
        print(
            f"Synthetische Sequenzen erzeugt: {len(sequences)} "
            f"(Längen {args.min_len}..{args.max_len}, Schritt {args.step})"
        )
    elif args.mode == "fasta":
        fasta_dir = Path(args.fasta_dir)
        if not fasta_dir.is_dir():
            raise NotADirectoryError(f"{fasta_dir} ist kein Verzeichnis.")
        sequences, lengths, _meta = collect_all_sequences(fasta_dir)
        print(
            f"FASTA-Sequenzen geladen: {len(sequences)} " f"aus Verzeichnis {fasta_dir}"
        )
    else:
        raise ValueError(f"Unbekannter Modus: {args.mode}")

    lengths = np.asarray(lengths, dtype=int)
    print(f"Gesamtanzahl Sequenzen: {len(sequences)}")
    print(f"Min-Länge: {lengths.min()}, Max-Länge: {lengths.max()}")
    return sequences, lengths


# ---------------------------------------------------------------------------
# Hauptlogik
# ---------------------------------------------------------------------------


def main():
    # GPU-Auswahl (nur Device 0 sichtbar machen)
    select_first_gpu(device_id=0)

    parser = argparse.ArgumentParser(
        description=(
            "Analyse von AgroNT-Embeddings für Sequenzen unterschiedlicher "
            "Kontextlänge mit globalen Statistiken, PCA/UMAP und "
            "SNP-spezifischen Distanz-/Cosine-Plots."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["synthetic", "fasta"],
        required=True,
        help="Eingabemodus: 'synthetic' für generierte SNP-Kontexte oder 'fasta' für Sequenzen aus FASTA-Dateien.",
    )

    # Parameter für synthetische Sequenzen
    parser.add_argument(
        "--min-len",
        type=int,
        default=6,
        help="Minimale Sequenzlänge (nur im Modus 'synthetic').",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=120,
        help="Maximale Sequenzlänge (nur im Modus 'synthetic').",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=6,
        help="Schrittweite der Längen (z.B. 6 -> 6,12,...). Nur im Modus 'synthetic'.",
    )
    parser.add_argument(
        "--num-per-length",
        type=int,
        default=1000,
        help="Anzahl Sequenzen pro Länge (nur im Modus 'synthetic').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Zufallsseed für die Sequenzgenerierung (nur im Modus 'synthetic').",
    )

    # Parameter für FASTA-Modus
    parser.add_argument(
        "--fasta-dir",
        type=str,
        default=None,
        help="Verzeichnis mit FASTA-Dateien (*.fa*), nur im Modus 'fasta' relevant.",
    )

    # Modell-Parameter
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
        "--max-positions",
        type=int,
        default=126,
        help="max_positions des Modells (Token-Länge).",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["seq", "cls"],
        default="seq",
        help="Pooling-Strategie: 'seq' = Sequenz-Token, 'cls' = CLS-Token.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batchgröße für die Inferenz.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="agront_outputs",
        help="Ausgabeverzeichnis für Plots (wird bei Bedarf angelegt).",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Sequenzen bereitstellen
    sequences, lengths = build_sequences_from_mode(args)

    # 2) Modell laden
    parameters, forward_fn, tokenizer, config = load_agront_model(
        max_positions=args.max_positions,
        model_name=args.model_name,
        layer_to_save=args.layer,
    )

    # 3) Embeddings berechnen
    rng = jax.random.PRNGKey(0)
    X = get_sequence_embeddings(
        sequences=sequences,
        parameters=parameters,
        forward_fn=forward_fn,
        tokenizer=tokenizer,
        rng_key=rng,
        layer_to_save=args.layer,
        pooling=args.pooling,
        batch_size=args.batch_size,
    )  # Shape: (N, hidden_dim)

    N, hidden_dim = X.shape
    print(f"\nFertige Embeddings-Shape: (N={N}, hidden_dim={hidden_dim})")

    # 4) Globale Cosine-/Distanz-Matrizen (nur Statistik)
    cos_mat = cosine_similarity_matrix(X)
    summarize_matrix("GLOBAL Sequenz-Embeddings Cosine-Similarity", cos_mat)

    dist_mat = euclidean_distance_matrix(X)
    summarize_matrix("GLOBAL Sequenz-Embeddings Euclidean Distance", dist_mat)

    # 5) SNP-spezifische Statistiken & Plots
    snp_stats = compute_snp_stats_by_length(
        sequences=sequences,
        lengths=lengths,
        embeddings=X,
    )

    print(
        "\nSNP-Paar-Statistiken pro Sequenzlänge "
        "(gleiche Flanken, SNP in der Mitte):"
    )
    for L in sorted(snp_stats.keys()):
        stats_L = snp_stats[L]
        print(
            f"  Länge {L:>3} bp: "
            f"n_pairs={stats_L['n_pairs']}, "
            f"mean_cosine={stats_L['mean_cosine']:.4f}, "
            f"mean_distance={stats_L['mean_distance']:.4f}"
        )

    plot_snp_stats_by_length(
        snp_stats=snp_stats,
        output_cosine_path=output_dir / "snp_mean_cosine_vs_length.png",
        output_distance_path=output_dir / "snp_mean_distance_vs_length.png",
    )

    # 6) PCA & UMAP auf allen Sequenz-Embeddings
    X_pca = pca_2d(X)
    X_umap = umap_2d(
        X,
        n_neighbors=5,
        min_dist=0.1,
        metric="cosine",
        random_state=0,
    )

    plot_pca(
        X_pca=X_pca,
        lengths=lengths,
        output_path=output_dir / "pca_sequence_embeddings.png",
    )

    plot_umap(
        X_umap=X_umap,
        lengths=lengths,
        output_path=output_dir / "umap_sequence_embeddings.png",
    )

    print("\nAnalyse abgeschlossen.")


# ---------------------------------------------------------------------------
# Logging: Terminal + log.txt gleichzeitig (Tee)
# ---------------------------------------------------------------------------

import sys


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


if __name__ == "__main__":
    log_file = open("log.txt", "w")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # alles gleichzeitig in Konsole und Logfile schreiben
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        main()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
