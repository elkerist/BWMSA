# BWMSA: BERT-Whitening MSA

BWMSA is a training-free pipeline for constructing protein multiple sequence
alignments. It applies a BERT-whitening transformation to ESMC-600M embeddings,
retrieves homolog candidates from UniRef90 via FAISS, then extends the
candidates into an MSA with `qjackhmmer`.

## Installation

```bash
git clone <this-repo>
cd BWMSA
conda env create -f environment.yml
conda activate esm
```

`qjackhmmer` must also be installed as a system binary (called as a subprocess
in step 5).

## Offline phase (once per database)

### 1. Preprocess UniRef90
```bash
python scripts/01_preprocess_database.py \
    --input ./data/uniref90.fasta \
    --filtered ./data/uniref90_minlen30.fasta \
    --output ./data/uniref90_filtered.fasta
```
Drops sequences shorter than 30 aa, truncates anything over 1024 aa.

### 2. Generate ESMC-600M embeddings
```bash
python scripts/02_build_embeddings.py
```

### 3. Build the whitened FAISS index
```bash
python scripts/03_build_index.py --emb_file ./data/protein_embeddings.npy
```
Writes `trained_index.faiss` and `whiten_params.npz`.

## Online phase (per query)

### 4. Homolog search
```bash
python run_homolog_search.py --query_fasta your_queries.fasta --output_dir results

# Default run (uses examples/input/example_query.fasta):
python run_homolog_search.py
```
For each query, writes the top-k UniRef90 candidates to
`results/candidates_<n>_<seq_id>.fasta`.

### 5. MSA generation
First split the multi-query FASTA into per-query `<n>.fa` files, where `<n>`
matches the index used in the candidate filenames from step 4:
```bash
python scripts/04_split_queries.py \
    --query_fasta your_queries.fasta \
    --output_dir  ./casp_query
```

Then run `qjackhmmer` against each query's candidate set:
```bash
python scripts/05_run_jackhmmer.py \
    --qjackhmmer      /path/to/qjackhmmer \
    --query_dir       ./casp_query \
    --candidates_dir  ./results \
    --output_dir      ./a3m_out
```

The `.a3m` files in `--output_dir` are the BWMSA MSAs.

## Optional tools

Under `scripts/optional/`:

- `cut_meff.py` + `batch_meff_gpu.py` — measure MSA depth (log Meff) on GPU.
- `a3m_combine.py` — stitch BWMSA's A3M onto ColabFold's UniRef + environmental
  A3Ms via hhfilter, for a combined MSA.

## File structure

```
BWMSA/
├── run_homolog_search.py
├── environment.yml
├── README.md
├── bwmsa/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── embedding.py
│   ├── indexing.py
│   └── search.py
├── scripts/
│   ├── 01_preprocess_database.py
│   ├── 02_build_embeddings.py
│   ├── 03_build_index.py
│   ├── 04_split_queries.py
│   ├── 05_run_jackhmmer.py
│   └── optional/
│       ├── cut_meff.py
│       ├── batch_meff_gpu.py
│       └── a3m_combine.py
├── examples/
│   └── input/
│       └── example_query.fasta
└── data/                            # created during setup (gitignored)
```

## Requirements

- Python 3.10
- PyTorch
- esm (EvolutionaryScale ESMC SDK)
- FAISS
- BioPython
- `qjackhmmer` system binary
- `hhfilter` from HH-suite (only for `a3m_combine.py`)
- See `environment.yml` for the full dependency list
