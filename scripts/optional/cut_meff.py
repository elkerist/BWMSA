"""Strip A3M insertion lowercase letters and collapse each sequence to a single line.

Required as a preprocessing step before batch_meff_gpu.py, which expects the
fasta-style A3M (one header line + one sequence line per record) so that
AlignIO.read can parse all sequences as the same fixed length.
"""

import argparse
import os


def process_a3m_to_oneline(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    a3m_files = [f for f in os.listdir(input_dir) if f.endswith(".a3m")]
    print(f"Processing {len(a3m_files)} files...")

    for filename in a3m_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, "r") as infile, open(output_path, "w") as outfile:
            first_record = True
            curr_seq = []

            for line in infile:
                line = line.strip()
                if not line:
                    continue

                if line.startswith(">"):
                    if not first_record:
                        outfile.write("".join(curr_seq) + "\n")
                    outfile.write(line + "\n")
                    curr_seq = []
                    first_record = False
                else:
                    # Drop lowercase A3M insertion residues so all rows have equal length.
                    clean_part = "".join(c for c in line if not c.islower())
                    curr_seq.append(clean_part)

            if curr_seq:
                outfile.write("".join(curr_seq) + "\n")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="A3M -> one-line FASTA-aligned preprocessor")
    parser.add_argument("--input_dir", required=True, help="Directory of input .a3m files")
    parser.add_argument("--output_dir", required=True, help="Directory for preprocessed .a3m files")
    args = parser.parse_args()
    process_a3m_to_oneline(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
