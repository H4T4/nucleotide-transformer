#!/usr/bin/env python3
"""
Wrapper für AgroNT:
- GPU-Auswahl
- Modell laden
- Sequenz-Embeddings berechnen (CLS- oder Sequenz-Token)
"""

import os
from typing import List, Tuple

import torch
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from nucleotide_transformer.pretrained import get_pretrained_model


def select_first_gpu(device_id: int = 0) -> None:
    """
    Setzt CUDA_VISIBLE_DEVICES auf die gewünschte GPU und
    zeigt an, wie viele GPUs PyTorch sieht.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print("CUDA_VISIBLE_DEVICES =", os.environ["CUDA_VISIBLE_DEVICES"])
    print("Anzahl sichtbarer CUDA Devices (PyTorch):", torch.cuda.device_count())


def load_agront_model(
    max_positions: int,
    model_name: str = "1B_agro_nt",
    layer_to_save: int = 12,
):
    """
    Lädt das AgroNT-Modell mit gewünschtem max_positions und Layer-Ausgabe.

    Wichtig:
    - embeddings_layers_to_save steuert nur, welche Hidden States im Output
      verfügbar sind (z.B. 'embeddings_12'), nicht welche Gewichte geladen werden.

    Rückgabe
    --------
    parameters, forward_fn, tokenizer, config
    """
    print(f"Lade Modell '{model_name}' mit max_positions={max_positions} ...")
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        embeddings_layers_to_save=(layer_to_save,),
        max_positions=max_positions,
    )
    # In vielen Implementierungen ist forward_fn schon transformiert.
    # Falls nicht, wäre hier hk.transform(forward_fn) nötig.
    # Wir prüfen pragmatisch auf 'apply'.
    if not hasattr(forward_fn, "apply"):
        forward_fn = hk.transform(forward_fn)

    return parameters, forward_fn, tokenizer, config


def get_sequence_embeddings(
    sequences: List[str],
    parameters,
    forward_fn,
    tokenizer,
    rng_key,
    layer_to_save: int = 12,
    pooling: str = "seq",  # "seq" = Sequenz-Token, "cls" = CLS-Token
    batch_size: int = 256,
) -> np.ndarray:
    """
    Berechnet Sequenz-Embeddings für eine Liste von Sequenzen.

    Aktuelles Tokenizer-Verhalten (wie bei dir beobachtet):
      model_input = ['<cls>', SEQ_AS_TOKEN, '<pad>', '<pad>', ...]
      -> Wir nehmen entweder:
         - CLS-Embedding (Index 0)  oder
         - Sequenz-Token-Embedding (Index 1)

    Parameter
    ---------
    sequences : List[str]
        Eingangsdaten (ACGT-Strings).
    pooling : str
        "cls"  -> Embedding an Token-Position 0
        "seq"  -> Embedding an Token-Position 1 (komplette Sequenz)
    batch_size : int
        Batch-Größe für die Inferenz.

    Rückgabe
    --------
    np.ndarray
        Array mit Shape (N, hidden_dim), N = len(sequences)
    """
    assert pooling in ("cls", "seq"), "pooling muss 'cls' oder 'seq' sein."

    all_embs: List[np.ndarray] = []
    n = len(sequences)
    print(f"Berechne Embeddings für {n} Sequenzen ...")

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_seqs = sequences[start:end]

        token_batch = tokenizer.batch_tokenize(batch_seqs)
        token_ids = [b[1] for b in token_batch]  # b = (token_string, token_ids, ...)
        tokens = jnp.asarray(token_ids, dtype=jnp.int32)

        # deterministischer Subkey
        subkey = jax.random.fold_in(rng_key, start)

        outs = forward_fn.apply(parameters, subkey, tokens)
        key = f"embeddings_{layer_to_save}"
        if key not in outs:
            raise KeyError(
                f"{key} nicht im Model-Output gefunden. Verfügbare Keys: {list(outs.keys())}"
            )
        emb = outs[key]  # (batch, max_positions, hidden_dim)

        if pooling == "cls":
            pooled = emb[:, 0, :]  # CLS
        else:
            pooled = emb[:, 1, :]  # Sequenz-Token

        all_embs.append(np.array(pooled))

    X = np.vstack(all_embs)  # (N, hidden_dim)
    print("Fertige Embeddings-Shape:", X.shape)
    return X
