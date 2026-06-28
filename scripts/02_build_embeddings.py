"""Encode each sequence in the preprocessed FASTA with ESMC-600M and stream to a single .npy.

The output array is memory-mapped, so the full embedding matrix never has to fit in RAM —
each row is written to disk as it is produced. Suitable for UniRef90-scale databases.
"""

import argparse

import numpy as np
from Bio import SeqIO

from bwmsa.embedding import embed_sequence, load_esmc


def main():
    parser = argparse.ArgumentParser(description="Generate ESMC-600M embeddings for the database")
    parser.add_argument(
        "--input_fasta",
        type=str,
        default="./data/uniref90_filtered.fasta",
        help="Preprocessed FASTA from 01_preprocess_database.py",
    )
    parser.add_argument(
        "--output_npy",
        type=str,
        default="./data/protein_embeddings.npy",
        help="Output .npy path (shape [N, 1152])",
    )
    parser.add_argument(
        "--flush_interval",
        type=int,
        default=10000,
        help="Flush the memmap and print progress every N sequences (default: 10000)",
    )
    args = parser.parse_args()

    print("Counting sequences...")
    n = sum(1 for _ in SeqIO.parse(args.input_fasta, "fasta"))
    if n == 0:
        raise ValueError(f"No sequences found in {args.input_fasta}")
    print(f"Total sequences: {n}")

    print("Loading ESMC model...")
    client = load_esmc()

    # Embed the first record to determine the embedding dimension, then allocate memmap.
    first_record = next(SeqIO.parse(args.input_fasta, "fasta"))
    first_emb = embed_sequence(client, str(first_record.seq))
    dim = first_emb.shape[0]
    print(f"Embedding dimension: {dim}")

    out = np.lib.format.open_memmap(
        args.output_npy, mode="w+", dtype=np.float32, shape=(n, dim)
    )
    out[0] = first_emb
    out.flush()

    for i, record in enumerate(SeqIO.parse(args.input_fasta, "fasta")):
        if i == 0:
            continue
        out[i] = embed_sequence(client, str(record.seq))
        if (i + 1) % args.flush_interval == 0:
            out.flush()
            print(f"Processed {i + 1}/{n} sequences", flush=True)

    out.flush()
    print(f"Saved embeddings of shape ({n}, {dim}) to {args.output_npy}")


if __name__ == "__main__":
    main()
