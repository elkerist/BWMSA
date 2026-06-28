"""BWMSA homolog search entry point (online phase).

Embeds each query sequence with ESMC-600M, whitens, and retrieves the top-k
nearest UniRef90 neighbours from the FAISS index. The candidate sequences for
each query are saved to {output_dir}/candidates_{n}_{seq_id}.fasta — those are
the inputs to scripts/05_run_jackhmmer.py.

Usage:
    # Default run (uses examples/input/example_query.fasta):
    python run_homolog_search.py

    # Custom query file and depth:
    python run_homolog_search.py --query_fasta my_queries.fasta --topk 1000000
"""

import argparse
import os
import time

from bwmsa.embedding import ProteinEmbeddingGenerator
from bwmsa.search import OptimizedSearcher


def main():
    parser = argparse.ArgumentParser(description="BWMSA homolog search")
    parser.add_argument(
        "--query_fasta",
        type=str,
        default="./examples/input/example_query.fasta",
        help="Query FASTA file path (default: ./examples/input/example_query.fasta)",
    )
    parser.add_argument(
        "--whiten_params",
        type=str,
        default="./data/whiten_params.npz",
        help="Whitening parameters file (default: ./data/whiten_params.npz)",
    )
    parser.add_argument(
        "--trained_index",
        type=str,
        default="./data/trained_index.faiss",
        help="FAISS index file (default: ./data/trained_index.faiss)",
    )
    parser.add_argument(
        "--database_fasta",
        type=str,
        default="./data/uniref90_filtered.fasta",
        help="UniRef90 (or other reference) FASTA matching the index rows",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="./data/sequence_index.npz",
        help="FASTA byte-offset index .npz (auto-created on first run)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for candidate files (default: ./results)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Query batch size; default = all queries at once",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1_000_000,
        help="Number of candidate sequences per query (default: 1,000,000)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("BWMSA homolog search")
    print("=" * 60)
    total_start = time.time()

    print("\nStep 1: Generate query sequence embeddings")
    print("-" * 40)
    embedder = ProteinEmbeddingGenerator(args.whiten_params)
    query_embeddings, query_identifiers = embedder.generate_embeddings(args.query_fasta)

    print("\nStep 2: Initialize FAISS searcher")
    print("-" * 40)
    searcher = OptimizedSearcher(args.trained_index, args.database_fasta, args.index_path)

    print("\nStep 3: Execute similarity search")
    print("-" * 40)
    num_queries = len(query_embeddings)
    batch_size = args.batch_size or num_queries
    print(
        f"Using {'default' if args.batch_size is None else 'custom'} batch size: {batch_size}"
        + (" (all query sequences)" if args.batch_size is None else "")
    )

    for batch_start in range(0, num_queries, batch_size):
        batch_end = min(batch_start + batch_size, num_queries)
        print(f"\nProcessing batch: queries {batch_start + 1}-{batch_end}/{num_queries}")

        batch_embeddings = query_embeddings[batch_start:batch_end]
        batch_identifiers = query_identifiers[batch_start:batch_end]

        distances, indices_batch = searcher.search_batch(batch_embeddings, topk=args.topk)

        for i, distance_row in enumerate(distances):
            print(f"Query {batch_identifiers[i]} similarity scores:", distance_row[:5])

        indices_list = [indices_batch[i] for i in range(len(batch_embeddings))]
        searcher.save_results_with_sequences(indices_list, batch_identifiers, args.output_dir)

    print("\n" + "=" * 60)
    print(f"All processing completed! Total time taken: {time.time() - total_start:.2f} seconds")
    print(f"Processed {num_queries} query sequences")
    print(f"Results saved in directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()


