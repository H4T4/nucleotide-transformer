Guter Punkt – jetzt haben wir drei schöne Skripte, aber sie müssen ja auch irgendwie „leben“ 🙂
Ich geh’s Schritt für Schritt durch.

---

## 1. Dateien anlegen

Lege dir z.B. ein Verzeichnis an:

```bash
mkdir agront_length_analysis
cd agront_length_analysis
```

Speichere dort die drei Dateien:

- `sequence_utils.py`
- `agront_model.py`
- `analysis_utils.py`
- `run_agront_analysis.py`

Achte darauf, dass die Dateinamen exakt so heißen.

---

## 2. Abhängigkeiten installieren (falls noch nicht passiert)

In deiner Conda-Umgebung (`agront` o.ä.):

```bash
conda activate agront  # falls nötig

pip install umap-learn matplotlib
# den Rest (jax, haiku, torch, nucleotide-transformer) hast du vermutlich schon,
# sonst:
# pip install "jax[cuda]" dm-haiku torch nucleotide-transformer
```

(Du kannst die fehlenden Pakete einfach nachinstallieren, falls Python sich beschwert.)

---

## 3. Skript für **synthetische Sequenzen** verwenden

Das Hauptskript ist: `run_agront_analysis.py`.

### Standardfall: genau dein bisheriges SNP-Setup

Im Projektordner:

```bash
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
  --output-dir agront_outputs_synthetic
```

Was das macht:

- **Sequenzgenerierung**:

  - Für Längen 6, 12, 18, …, 120
  - Pro Länge 1000 Sequenzen
  - SNP in der Mitte, 250 Sequenzen pro Mittel-Nukleotid (A/C/G/T)

- **Modell**:

  - lädt `1B_agro_nt` mit `max_positions=126` und speichert `embeddings_12`

- **Embeddings**:

  - nimmt das **Sequenz-Token** (Token-Index 1) als Sequenz-Embedding (`--pooling seq`)

- **Analysen**:

  - Cosine-Similarity-Matrix (auf allen Sequenz-Embeddings)
  - PCA (2D)
  - UMAP (2D)
  - Cosine-Similarity zur Referenzlänge 60 bp (oder längste, falls keine 60er)

- **Outputs** (im Ordner `agront_outputs_synthetic`):

  - `pca_sequence_embeddings.png`
  - `umap_sequence_embeddings.png`
  - `cosine_vs_length.png`

Die Plots kannst du dir dann z.B. in VSCode oder im Dateimanager anschauen.

---

## 4. Skript für **FASTA-Sequenzen** verwenden

Wenn du stattdessen Sequenzen aus Dateien nehmen willst:

### 4.1 FASTA-Verzeichnis vorbereiten

Beispiel:

```bash
mkdir fasta_files
# lege dort Dateien wie:
#   chr1_sequences.fa
#   context_6bp.fa
#   context_12bp.fa
# etc.
```

Jede Datei im FASTA-Format, z.B.:

```text
>seq1
ACGTACGTACGT
>seq2
TTTGGGCCC
...
```

### 4.2 Analyse starten

```bash
python run_agront_analysis.py \
  --mode fasta \
  --fasta-dir ./fasta_files \
  --model-name 1B_agro_nt \
  --layer 12 \
  --max-positions 126 \
  --pooling seq \
  --output-dir agront_outputs_fasta
```

Was hier passiert:

- `sequence_utils.collect_all_sequences` liest alle `*.fa*` im Ordner.
- Alle Sequenzen kommen in eine große Liste, Längen werden erfasst.
- Modell wird geladen wie oben.
- Sequenz-Embeddings werden berechnet (`pooling=seq`).
- Cosine-Matrix, PCA, UMAP, Cosine-vs-Länge werden berechnet und geplottet.

---

## 5. Was, wenn du das lieber im **Jupyter Notebook** verwenden willst?

Dann kannst du die Funktionen direkt importieren.

Im Notebook (im selben Ordner wie die `.py`-Files):

```python
from sequence_utils import generate_snp_variant_sequences, collect_all_sequences
from agront_model import select_first_gpu, load_agront_model, get_sequence_embeddings
from analysis_utils import (
    cosine_similarity_matrix, cosine_to_reference, summarize_matrix,
    pca_2d, umap_2d, plot_pca, plot_umap, plot_cosine_vs_length
)
```

Beispiel: nur Embeddings für eine Länge erzeugen:

```python
# 1) GPU auswählen
select_first_gpu(0)

# 2) Sequenzen generieren
sequences, lengths = generate_snp_variant_sequences(
    min_len=60, max_len=60, step=6, per_length=1000, seed=0
)

# 3) Modell laden
parameters, forward_fn, tokenizer, config = load_agront_model(
    max_positions=126, model_name="1B_agro_nt", layer_to_save=12
)

import jax
rng = jax.random.PRNGKey(0)

# 4) Embeddings berechnen
X = get_sequence_embeddings(
    sequences=sequences,
    parameters=parameters,
    forward_fn=forward_fn,
    tokenizer=tokenizer,
    rng_key=rng,
    layer_to_save=12,
    pooling="seq",
    batch_size=256,
)

X.shape  # (1000, hidden_dim)
```

Du kannst also dasselbe Backend (Generierung + Modell + Embeddings + Analysen) sowohl in der Shell als auch im Notebook nutzen.

---

## 6. Kleiner Spickzettel

- **Synthetic-Modus** (deine SNP-Kontexte):

  ```bash
  python run_agront_analysis.py --mode synthetic ...
  ```

- **FASTA-Modus**:

  ```bash
  python run_agront_analysis.py --mode fasta --fasta-dir ./fasta_files ...
  ```

Wichtige Parameter, die du später leicht anpassen kannst:

- `--min-len`, `--max-len`, `--step`, `--num-per-length` (für synthetische Experimente)
- `--max-positions` (Token-Fenster des Modells, aktuell 126, passend für bis 120 nt)
- `--pooling cls` vs. `--pooling seq` (CLS-Embedding vs. Sequenz-Token-Embedding)
- `--output-dir` (damit du verschiedene Runs getrennt speichern kannst)

Wenn du magst, können wir als nächstes einen „minimalen“ Run definieren (z.B. nur 6, 60, 120 bp, weniger Sequenzen), um Dinge schneller zu testen, bevor du mit „vollen“ 1000 Sequenzen pro Länge arbeitest.
