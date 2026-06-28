"""Split a multi-query FASTA into per-query <n>.fa files for 05_run_jackhmmer.py.

The numbering must match the `<n>` used by run_homolog_search.py when it wrote
candidates_<n>_<seq_id>.fasta — that is, the 1-based index of each record in
the input FASTA. 05_run_jackhmmer.py pairs <n>.fa against candidates_<n>_*.fasta
via that shared index.
"""

import argparse
import os

from Bio import SeqIO


def main():
    parser = argparse.ArgumentParser(description="Split multi-query FASTA into per-query <n>.fa files")
    parser.add_argument("--query_fasta", required=True, help="Multi-query FASTA (same file you passed to run_homolog_search.py)")
    parser.add_argument("--output_dir", required=True, help="Directory to write per-query <n>.fa files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for num, record in enumerate(SeqIO.parse(args.query_fasta, "fasta"), start=1):
        out_path = os.path.join(args.output_dir, f"{num}.fa")
        with open(out_path, "w") as f:
            SeqIO.write(record, f, "fasta")
        print(f"Wrote {out_path}  (id={record.id})")


if __name__ == "__main__":
    main()
