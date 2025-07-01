# 🚀 OrthologTransformer: Smarter Gene Conversion Across Species!

OrthologTransformer is a Transformer-based deep learning model that lets you **intelligently redesign genes** to work in different species. Instead of just swapping synonymous codons, it learns from **real ortholog pairs** to capture evolutionary patterns, non-synonymous changes, indels, and more. Think of it as AI-assisted molecular evolution—with a purpose!

---

## ✨ Features

* 🧬 Converts genes at the codon and amino acid level
* 🧠 Learns evolutionary patterns from real orthologs
* 🎯 Adapt genes to specific species with context-aware predictions
* 🔁 Includes pretrained models (alignment-based and alignment-free!)
* 🧪 Validated in the lab with PETase activity in *B. subtilis*
* 🔧 Built-in MCTS optimization for GC content and RNA structure

---

## 📦 Requirements

Install everything with:

```bash
pip install -r requirements.txt
```

And here's your `requirements.txt`:

```txt
torch>=2.0
biopython>=1.85
numpy>=1.26
pandas>=2.2
scipy>=1.13
schedulefree==1.4
ViennaRNA==2.6.4
```


---

## 🔍 Inference (Predict and Reconstruct Sequences)

Run the following command to predict a gene sequence adapted from a source species to a target species:

```bash
python run_inference.py \
  --model_input results/train/model_best.pt \
  --ortholog_files_test ECOLI_IDESA.fasta \
  --result_folder results/test
```

This generates predicted sequences transforming from `IDESA` (source) to `ECOLI` (target).

---

### 🧾 FASTA Input Format

```fasta
>ECOLI00000


>IDESA05470
ATGAACTTTCCCCGCGCTTCCCGCCTGATGCAGGCCGCCGTTCTCGGCGGGCTGATGGCC...
```

* Name the FASTA file as `<TARGET>_<SOURCE>.fasta`, e.g., `ECOLI_IDESA.fasta`
* The first sequence (with no bases) is the target to be predicted. The second sequence is the source to be used for inference.
* Lookup valid species codes here 👉 [OMA species list](https://omabrowser.org/All/oma-species.txt)
* The `target` sequence should normally be **left blank**
* If you use `--edition_fasta`, the `target` contains a **starting draft sequence**

---

### 📂 Output Files

After running the script, the following files will appear in your `--result_folder`:

| File                        | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| `args_test.json`            | Parameters used for the inference run                        |
| `completed_sequences.fasta` | DNA sequences (predicted + source), in standard FASTA format |
| `completed_proteins.fasta`  | Protein sequences translated from the DNA output             |

---

### 🧬 Output Example (`completed_sequences.fasta`)

```fasta
>ECOLI00000
ATGAACAAAACACTCTGCTCTCTCTTTCTGGTCACAGCCAGCTTGATTACACCAGCAAGC...

>IDESA05470
ATGAACTTTCCCCGCGCTTCCCGCCTGATGCAGGCCGCCGTTCTCGGCGGGCTGATGGCC...
```

The first sequence is the predicted gene in the target species (`ECOLI`).
The second is the original input sequence from the source species (`IDESA`).
Each entry is in standard FASTA format with no blank lines between them.

---

### ✏️ With Edition Mode

```bash
python run_inference.py \
  --model_input results/train/model_best.pt \
  --ortholog_files_test ECOLI_IDESA.fasta \
  --edition_fasta \
  --result_folder results/test_edit
```

Let the model refine your initial output!



### 🚀 Beam Search Decoding

To explore multiple high-quality candidate sequences in a single run, activate beam search:

```bash
python run_inference.py \
  --model_input results/train/model_best.pt \
  --ortholog_files_test ECOLI_IDESA.fasta \
  --result_folder results/test_beam \
  --use_beam \            # turn on beam search
  --beam_size 8 \         # width of the beam (≥1)
  --num_return_sequences 4    # how many sequences to keep
```



### 🔬 With MCTS Optimization

```bash
python run_inference.py \
  --model_input results/train/model_best.pt \
  --ortholog_files_test ECOLI_IDESA.fasta \
  --result_folder results/test_mcts \
  --mcts
```

* No tuning needed
* Targets GC ≈ 0.365
* Minimizes mRNA folding energy (MFE)

---

## 🧠 Pretrained Models

| Model                | Description                | Link                                         |
| -------------------- | -------------------------- | -------------------------------------------- |
| `unaligned_model.pt` | Trained without alignments(SOTA) | [📥 Download](https://drive.google.com/file/d/1fJeTPh8MnH8fe_UA0SPxxXTCKouxcy3j/view?usp=drive_link) |
| `aligned_model.pt`   | Trained with alignments    | [📥 Download](https://drive.google.com/file/d/10sq137YwEhmr3OBw_Klt-Wd38tmmeHsV/view?usp=drive_link) |

---

## ⚡ GPU Support

OrthologTransformer flies on GPU (CUDA). Use one if you can!

---

## 🏋️‍♀️ Training From Scratch

```bash
python train.py \
  --ortholog_files_train data/train/*.fasta \
  --result_folder results/train \
  --num_epochs 100 \
  --batch_size 32
```

Each training file should be named like this:

```
groupXXXXX_TARGET_SOURCE.fasta
```

Where:

* `groupXXXXX` = ortholog group ID
* `TARGET` = target species
* `SOURCE` = original species

Example:

```
group00001_ECOLI_IDESA.fasta  → converts IDESA ➜ ECOLI
```

### FASTA Format for Training

```fasta
>target_species
ATGCCC...
>source_species
ATGGCC...
```

✅ Both sequences **must** be provided
✅ Species names must match OMA IDs

---

## 🔁 Fine-Tuning on New Pairs

```bash
python train.py \
  --model_input pretrained/aligned_model.pt \
  --ortholog_files_train data/finetune_pairs/*.fasta \
  --result_folder results/finetune \
  --num_epochs 20
```

Use this to adapt the model to new species pairs or functional domains!

---

## 📊 Datasets

| Dataset | Description | Link |
| ------- | ----------- | ---- |
| **Training Data** | 4.97M ortholog pairs from 2,138 bacterial species | [📥 Download]https://drive.google.com/file/d/1lmUVb8jQYdGNxeAC73Xeg7h-17BQVntO/view?usp=drive_link) |
| **Species-pair Data** | Test sets and fine-tuning data for specific species pairs | [📥 Download](https://drive.google.com/file/d/1Qba8jXkWleWmWBULtqKBPBWw7C7KxfdG/view?usp=drive_link) |

All datasets use standard FASTA format with [OMA species IDs](https://omabrowser.org/All/oma-species.txt).



---

## 📚 Citation

```
Manato Akiyama, Motohiko Tashiro, Ying Huang, Mika Uehara, Taiki Kanzaki, Mitsuhiro Itaya, Masakazu Kataoka, Kenji Miyamoto, and Yasubumi Sakakibara (2025). Generative AI-driven artificial DNA design for enhancing inter-species gene activation and enzymatic degradation of PET. bioRxiv. https://doi.org/10.1101/2025.05.08.652991
```

---

## 📬 Contact

**Manato Akiyama**
Kitasato University, School of Frontier Engineering
📧 [akiyama.manato@kitasato-u.ac.jp](mailto:akiyama.manato@kitasato-u.ac.jp)
