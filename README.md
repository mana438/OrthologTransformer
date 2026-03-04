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


## 📄 Paper

Our work has been published in **Nature Communications**:

**Cross-species gene redesign leveraging ortholog information and generative modeling**  
Manato Akiyama, Motohiko Tashiro, Ying Huang, Mika Uehara, Taiki Kanzaki, Mitsuhiro Itaya, Masakazu Kataoka, Kenji Miyamoto, Yasubumi Sakakibara  

🔗 https://www.nature.com/articles/s41467-026-69966-0

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

> **Note:**  
> If `--result_folder` is not specified, the program automatically creates a folder such as `results/<timestamp>/`.  
> If you do specify a folder (e.g., `--result_folder results/test`), it will also be created automatically if it does not exist.

---

### 🧾 FASTA Input Format

```fasta
>ECOLI00000


>IDESA05470
ATGAACTTTCCCCGCGCTTCCCGCCTGATGCAGGCCGCCGTTCTCGGCGGGCTGATGGCC...
```

* Name the FASTA file as `<TARGET>_<SOURCE>.fasta`, e.g., `ECOLI_IDESA.fasta`
* Lookup valid species codes here 👉 [OMA species list](https://omabrowser.org/All/oma-species.txt)
* The first sequence (with no bases) is the target to be predicted. The second sequence is the source to be used for inference.
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


⚠️ **Important note on decoding for `unaligned_model.pt`**

Because `unaligned_model.pt` is trained without alignment constraints, it allows a broader range of sequence modifications, including non-synonymous substitutions and indels.  
While this increases the expressive power of the model, it also expands the search space and may result in structurally unstable sequences.  
To improve generation robustness and structural consistency, the use of **beam search decoding** is strongly recommended.

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
| **Training Data** | 4.97M ortholog pairs from 2,138 bacterial species | [📥 Download](https://drive.google.com/file/d/1SnmZ0QwvVWUq-X59xtgfwtRR7h1NmXFB/view?usp=drive_link) |
| **Large-scale Benchmark Test Set (Fig.3)** | Test dataset used for the large-scale benchmark across 45 bacterial species (Fig.3 in the paper) | [📥 Download](https://drive.google.com/file/d/1dp_ZV8pZ_u0G0EmQro5KTn8ucog_MVhi/view?usp=drive_link) |
| **Species-pair Data** | Test sets and fine-tuning data for specific species pairs (Table 1 experiments) | [📥 Download](https://drive.google.com/file/d/1Qba8jXkWleWmWBULtqKBPBWw7C7KxfdG/view?usp=drive_link) |

All datasets use standard FASTA format with [OMA species IDs](https://omabrowser.org/All/oma-species.txt).


---
## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@article{akiyama2026orthologtransformer,
  title={Cross-species gene redesign leveraging ortholog information and generative modeling},
  author={Akiyama, Manato and Tashiro, Motohiko and Huang, Ying and Uehara, Mika and Kanzaki, Taiki and Itaya, Mitsuhiro and Kataoka, Masakazu and Miyamoto, Kenji and Sakakibara, Yasubumi},
  journal={Nature Communications},
  year={2026},
  volume={17},
  pages={2120},
  doi={10.1038/s41467-026-69966-0}
}
```

---

## 📬 Contact

**Manato Akiyama**
Kitasato University, School of Frontier Engineering
📧 [akiyama.manato@kitasato-u.ac.jp](mailto:akiyama.manato@kitasato-u.ac.jp)
