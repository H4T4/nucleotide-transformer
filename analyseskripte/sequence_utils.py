#!/usr/bin/env python3
"""
Utilities für Sequenzgenerierung und FASTA-Handling.

- generate_snp_variant_sequences: synthetische SNP-Kontext-Sequenzen
- read_fasta_sequences / collect_all_sequences: FASTA-Dateien einlesen
"""

from pathlib import Path
from typing import Dict, List, Tuple
import random


NUCLEOTIDES = ["A", "C", "G", "T"]


def generate_snp_variant_sequences(
    min_len: int = 6,
    max_len: int = 120,
    step: int = 6,
    per_length: int = 1000,
    seed: int = 0,
) -> Tuple[List[str], List[int]]:
    """
    Generiert zufällige Sequenzen mit einem SNP in der Mitte für mehrere Längen.

    Für jede Länge L in [min_len, max_len] mit Schrittweite step:
      - es werden per_length Sequenzen erzeugt (ungefähr; per_length sollte
        durch 4 teilbar sein)
      - es werden zunächst T = per_length // 4 "Template"-Sequenzen erzeugt
        (vollständige zufällige Sequenz)
      - zu jedem Template werden 4 Varianten erzeugt, bei denen NUR die Base
        in der Mitte auf A / C / G / T gesetzt wird
        -> gleiche Flanken, unterschiedliche Mittelbase

    Rückgabe
    --------
    sequences : List[str]
        Alle generierten Sequenzen.
    lengths : List[int]
        Gleiche Länge wie sequences, enthält jeweils die Nukleotidlänge.
    """
    sequences: List[str] = []
    lengths: List[int] = []

    for L in range(min_len, max_len + 1, step):
        random.seed(seed + L)

        # wie viele Templates pro Länge? (4 Varianten pro Template)
        templates_per_length = per_length // len(NUCLEOTIDES)
        remainder = per_length % len(NUCLEOTIDES)
        if remainder != 0:
            # Wir ignorieren den Rest, damit exakt 4 Varianten pro Template entstehen.
            print(
                f"Warnung: per_length={per_length} ist nicht durch 4 teilbar. "
                f"Für Länge {L} werden nur {templates_per_length * 4} Sequenzen erzeugt."
            )

        mid = L // 2

        for _ in range(templates_per_length):
            # zufällige Template-Sequenz
            template = [random.choice(NUCLEOTIDES) for _ in range(L)]

            # 4 Varianten mit identischen Flanken, aber unterschiedlicher Base in der Mitte
            for nuc in NUCLEOTIDES:
                seq_list = template.copy()
                seq_list[mid] = nuc
                seq = "".join(seq_list)
                sequences.append(seq)
                lengths.append(L)

    return sequences, lengths


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
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current).upper())
                    current = []
            else:
                current.append(line)
        if current:
            sequences.append("".join(current).upper())

    return sequences


def collect_all_sequences(
    fasta_dir: Path,
) -> Tuple[List[str], List[int], Dict[int, Tuple[str, int]]]:
    """
    Liest alle FASTA-Dateien in einem Verzeichnis und gibt Sequenzen + Längen zurück.

    Parameter
    ---------
    fasta_dir : Path
        Verzeichnis, in dem nach FASTA-Dateien gesucht wird (*.fa*).

    Rückgabewert
    ------------
    sequences : List[str]
        Alle Sequenzen aus allen Dateien.
    lengths : List[int]
        Länge jeder Sequenz (in Nukleotiden).
    index_meta : Dict[int, (filename, length)]
        Mapping von globalem Index -> (Dateiname, Länge).
    """
    fasta_files = sorted(list(fasta_dir.glob("*.fa*")))
    if not fasta_files:
        raise FileNotFoundError(f"Keine FASTA-Dateien in {fasta_dir} gefunden.")

    sequences: List[str] = []
    lengths: List[int] = []
    index_meta: Dict[int, Tuple[str, int]] = {}

    idx = 0
    for fasta_file in fasta_files:
        seqs = read_fasta_sequences(fasta_file)
        if not seqs:
            print(f"Warnung: Keine Sequenzen in {fasta_file} gefunden.")
            continue
        for s in seqs:
            sequences.append(s)
            L = len(s)
            lengths.append(L)
            index_meta[idx] = (fasta_file.name, L)
            idx += 1

    if not sequences:
        raise RuntimeError(
            "Es wurden zwar FASTA-Dateien gefunden, aber keine Sequenzen."
        )

    return sequences, lengths, index_meta
