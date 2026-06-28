"""FASTA preprocessing: filter by minimum length and truncate to maximum length."""

from Bio import SeqIO


def filter_fasta_by_length(input_path, output_path, min_length=30):
    """Drop records shorter than min_length residues."""
    kept = 0
    total = 0
    with open(output_path, "w") as out:
        for record in SeqIO.parse(input_path, "fasta"):
            total += 1
            if len(record.seq) >= min_length:
                SeqIO.write(record, out, "fasta")
                kept += 1
    print(f"Filtered {input_path}: kept {kept}/{total} sequences (>= {min_length} aa)")


def truncate_long_sequences(input_path, output_path, max_length=1024):
    """Truncate records longer than max_length to their first max_length residues."""
    truncated = 0
    total = 0
    with open(output_path, "w") as out:
        for record in SeqIO.parse(input_path, "fasta"):
            total += 1
            if len(record.seq) > max_length:
                record.seq = record.seq[:max_length]
                truncated += 1
            SeqIO.write(record, out, "fasta")
    print(f"Processed {input_path}: truncated {truncated}/{total} sequences (> {max_length} aa)")
