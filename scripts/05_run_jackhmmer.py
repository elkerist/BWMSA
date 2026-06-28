"""Step 2: batch-run qjackhmmer over each query against its BWMSA candidate database.

For each query <id>.fa in --query_dir, finds the matching candidates file in
--candidates_dir (pattern: candidates_<id>_*.fasta as written by run_homolog_search.py)
and runs `qjackhmmer -N 3 -B <output> query db`, writing the .a3m to --output_dir.
"""

import argparse
import glob
import os
import subprocess


def run_qjackhmmer(query_file, database_file, output_file, qjackhmmer_path, n_iter=3):
    cmd = [
        qjackhmmer_path,
        "-N", str(n_iter),
        "-B", output_file,
        query_file,
        database_file,
    ]
    print(f"Processing: {query_file} -> {output_file}")
    print(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Success: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {query_file}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def find_matching_database(query_id, candidates_dir):
    pattern = os.path.join(candidates_dir, f"candidates_{query_id}_*.fasta")
    matching = glob.glob(pattern)
    return matching[0] if matching else None


def main():
    parser = argparse.ArgumentParser(description="Batch qjackhmmer for BWMSA candidate sets")
    parser.add_argument("--qjackhmmer", required=True,
                        help="Path to qjackhmmer binary")
    parser.add_argument("--query_dir", required=True,
                        help="Directory of per-query .fa files (one query each, named <id>.fa)")
    parser.add_argument("--candidates_dir", required=True,
                        help="Directory of candidate FASTAs from run_homolog_search.py")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for output .a3m files")
    parser.add_argument("--n_iter", type=int, default=3,
                        help="JackHMMER iterations (default: 3)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    query_files = sorted(glob.glob(os.path.join(args.query_dir, "*.fa")))
    if not query_files:
        print(f"No .fa files found in {args.query_dir}")
        return
    print(f"Found {len(query_files)} query files")

    success = 0
    failed = 0
    for query_file in query_files:
        query_id = os.path.splitext(os.path.basename(query_file))[0]
        database_file = find_matching_database(query_id, args.candidates_dir)
        if not database_file:
            print(f"Warning: no candidate database for {query_file}")
            failed += 1
            continue

        output_file = os.path.join(args.output_dir, f"{query_id}.a3m")
        if os.path.exists(output_file):
            print(f"Skipping: {output_file} already exists")
            continue

        if run_qjackhmmer(query_file, database_file, output_file, args.qjackhmmer, args.n_iter):
            success += 1
        else:
            failed += 1
        print("-" * 50)

    print(f"\nDone! Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()
