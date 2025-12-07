#!/usr/bin/env python3
"""
Gesamt-Skript für:
1. Sequenzgenerierung ODER FASTA-Import
2. Laden des AgroNT-Modells
3. Berechnung von Sequenz-Embeddings (letzter Layer)
4. Analysen:
   - Cosine-Similarity-Matrix
   - PCA (2D)
   - UMAP (2D)
   - Cosine-Similarity zur Referenzlänge (z.B. 60 bp)
5. Speichern der Plots

Beispielaufrufe:

# Synthetische Sequenzen (6..120 bp, Schritt 6, 1000 pro Länge)
python run_agront_analysis.py \
  --mode synthetic \
  --min-len 6 --max-len 120 --step 6 --num-per-length 1000 \
  --model-name 1B_agro_nt \
  --layer 12 \
  --max-positions 126 \
  --output-dir agront_outputs_synthetic

# FASTA-Sequenzen aus Verzeichnis
python run_agront_analysis.py \
  --mode fasta \
  --fasta-dir /pfad/zu/fasta_files \
  --model-name 1B_agro_nt \
  --layer 12 \
  --max-positions 126 \
  --output-dir agront_outputs_fasta
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import jax

from sequence_utils import (
    generate_snp_variant_sequences,
    collect_all_sequences,
)
from agront_model import (
    select_first_gpu,
    load_agront_model,
    get_sequence_embeddings,
)
from analysis_utils import (
    cosine_similarity_matrix,
    cosine_to_reference,
    euclidean_distance_matrix,  # NEU
    euclidean_to_reference,  # NEU
    summarize_matrix,
    pca_2d,
    umap_2d,
    plot_pca,
    plot_umap,
    plot_cosine_vs_length,
    plot_distance_vs_length,  # NEU
)


def build_sequences_from_mode(args) -> Tuple[List[str], np.ndarray]:
    """
    Erzeugt oder lädt Sequenzen je nach args.mode und
    gibt (sequences, lengths_array) zurück.
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
        lengths_arr = np.array(lengths, dtype=int)
        return sequences, lengths_arr

    elif args.mode == "fasta":
        if args.fasta_dir is None:
            raise ValueError("--fasta-dir ist für mode 'fasta' erforderlich.")
        fasta_dir = Path(args.fasta_dir)
        if not fasta_dir.is_dir():
            raise NotADirectoryError(f"{fasta_dir} ist kein Verzeichnis.")
        sequences, lengths, index_meta = collect_all_sequences(fasta_dir)
        print(f"{len(sequences)} Sequenzen aus FASTA-Dateien geladen.")
        print("Beispiele (Index: Datei, Länge):")
        for i in range(min(5, len(index_meta))):
            fname, L = index_meta[i]
            print(f"  {i}: Datei={fname}, Länge={L}")
        lengths_arr = np.array(lengths, dtype=int)
        return sequences, lengths_arr

    else:
        raise ValueError(f"Unbekannter mode: {args.mode}")


def main():
    select_first_gpu(device_id=0)

    parser = argparse.ArgumentParser(
        description="AgroNT-Analyse von Sequenz-Embeddings für synthetische oder FASTA-Sequenzen."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["synthetic", "fasta"],
        default="synthetic",
        help="synthetic: generierte SNP-Kontext-Sequenzen; fasta: Sequenzen aus FASTA-Dateien.",
    )

    # Synthetic-Parameter
    parser.add_argument(
        "--min-len", type=int, default=6, help="Minimale Sequenzlänge (synthetic)."
    )
    parser.add_argument(
        "--max-len", type=int, default=120, help="Maximale Sequenzlänge (synthetic)."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=6,
        help="Schrittweite der Sequenzlängen (synthetic).",
    )
    parser.add_argument(
        "--num-per-length",
        type=int,
        default=1000,
        help="Anzahl Sequenzen pro Länge (synthetic).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random-Seed für Sequenzgenerierung."
    )

    # FASTA-Parameter
    parser.add_argument(
        "--fasta-dir",
        type=str,
        default=None,
        help="Verzeichnis mit FASTA-Dateien (*.fa*), nur für mode=fasta.",
    )

    # Modellparameter
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
        help="max_positions für das Modell (Token-Länge im Modell).",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["cls", "seq"],
        default="seq",
        help="Welches Token-Embedding als Sequenz-Embedding verwendet wird: 'cls' oder 'seq'.",
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

    # 1) Sequenzen bauen / laden
    sequences, lengths = build_sequences_from_mode(args)
    N = len(sequences)
    print(f"Gesamtanzahl Sequenzen: {N}")
    print(f"Min-Länge: {lengths.min()}, Max-Länge: {lengths.max()}")

    # 2) Modell laden
    parameters, forward_fn, tokenizer, config = load_agront_model(
        max_positions=args.max_positions,
        model_name=args.model_name,
        layer_to_save=args.layer,
    )

    # Fixer Random Key (deterministische Inferenz)
    rng = jax.random.PRNGKey(0)

    # 3) Sequenz-Embeddings berechnen (Shape: (N, hidden_dim))
    X = get_sequence_embeddings(
        sequences=sequences,
        parameters=parameters,
        forward_fn=forward_fn,
        tokenizer=tokenizer,
        rng_key=rng,
        layer_to_save=args.layer,
        pooling=args.pooling,
        batch_size=512,
    )

    # Cosine-Similarity-Matrix
    cos_mat = cosine_similarity_matrix(X)
    summarize_matrix("GLOBAL Sequenz-Embeddings Cosine-Similarity", cos_mat)

    # Euklidische Distanz-Matrix (optional – groß!)
    dist_mat = euclidean_distance_matrix(X)
    summarize_matrix("GLOBAL Sequenz-Embeddings Euclidean Distance", dist_mat)

    # 5) Referenzsequenz bestimmen: bevorzugt Länge 60, sonst längste Sequenz
    target_length = 60
    candidate_indices = np.where(lengths == target_length)[0]
    if len(candidate_indices) > 0:
        ref_idx = int(candidate_indices[0])
    else:
        ref_idx = int(np.argmax(lengths))
        target_length = int(lengths[ref_idx])
        print(
            f"\nKeine Sequenz mit exakt 60 bp gefunden – "
            f"verwende Länge {target_length} bp als Referenz."
        )

    cos_to_ref = cosine_to_reference(X, ref_idx=ref_idx)
    dist_to_ref = euclidean_to_reference(X, ref_idx=ref_idx)

    # 6) PCA & UMAP berechnen
    X_pca = pca_2d(X)
    X_umap = umap_2d(
        X,
        n_neighbors=5,
        min_dist=0.1,
        metric="cosine",
        random_state=0,
    )

    # 7) Plots erstellen und speichern
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
    plot_cosine_vs_length(
        lengths=lengths,
        cos_to_ref=cos_to_ref,
        ref_length=target_length,
        output_path=output_dir / "cosine_vs_length.png",
    )

    plot_distance_vs_length(
        lengths=lengths,
        dist_to_ref=dist_to_ref,
        ref_length=target_length,
        output_path=output_dir / "distance_vs_length.png",
    )

    print("\nAnalyse abgeschlossen.")


if __name__ == "__main__":
    main()
