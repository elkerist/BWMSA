"""Filter the UniRef90 FASTA: drop sequences shorter than 30 aa, truncate any over 1024 aa."""

import argparse

from bwmsa.preprocessing import filter_fasta_by_length, truncate_long_sequences


def main():
    parser = argparse.ArgumentParser(description="Filter and truncate the reference FASTA")
    parser.add_argument("--input", required=True, help="Raw UniRef90 FASTA")
    parser.add_argument("--filtered", required=True, help="Intermediate path after length filter")
    parser.add_argument("--output", required=True, help="Final path after truncation")
    parser.add_argument("--min_length", type=int, default=30)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    filter_fasta_by_length(args.input, args.filtered, args.min_length)
    truncate_long_sequences(args.filtered, args.output, args.max_length)


if __name__ == "__main__":
    main()
