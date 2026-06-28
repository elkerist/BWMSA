"""Build the whitened FAISS index from the embedding .npy produced by 02_build_embeddings.py."""

import argparse

from bwmsa.indexing import build_index


def main():
    parser = argparse.ArgumentParser(description="Build whitened FAISS index from embedding .npy")
    parser.add_argument(
        "--emb_file",
        type=str,
        required=True,
        help="Pre-computed embedding .npy ([N, D] float32, expected D=1152 for ESMC-600M)",
    )
    parser.add_argument(
        "--index_out",
        type=str,
        default="./data/trained_index.faiss",
        help="Output path for the FAISS index (default: ./data/trained_index.faiss)",
    )
    parser.add_argument(
        "--params_out",
        type=str,
        default="./data/whiten_params.npz",
        help="Output path for whitening params (default: ./data/whiten_params.npz)",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=512,
        help="Reserved whitened dimension (default: 512)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1_000_000,
        help="Rows per streaming batch (default: 1,000,000)",
    )
    args = parser.parse_args()

    build_index(
        emb_file=args.emb_file,
        index_out=args.index_out,
        params_out=args.params_out,
        n_components=args.n_components,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
