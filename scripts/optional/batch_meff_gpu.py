"""GPU-accelerated effective sequence count (Meff) computation over a directory of A3M files.

Meff = Σ 1 / m_a, where m_a counts the sequences sharing ≥ identity_threshold
fraction of matching positions with sequence a.

Inputs must already be preprocessed via cut_meff.py (one-line FASTA-aligned),
since this script uses AlignIO.read which requires equal-length sequences.
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
from Bio import AlignIO


def calculate_meff_gpu(a3m_path, identity_threshold=0.8, batch_size=2):
    """Compute Meff and log Meff for one A3M on GPU. batch_size trades VRAM for speed."""
    try:
        alignment = AlignIO.read(a3m_path, "fasta")
        msa_np = np.stack(
            [np.frombuffer(str(rec.seq).encode(), dtype=np.uint8) for rec in alignment]
        )
        msa_tensor = torch.from_numpy(msa_np).cuda()
        num_seqs, seq_len = msa_tensor.shape
        threshold_match = identity_threshold * seq_len
        weights = torch.zeros(num_seqs, device="cuda")

        for i in range(0, num_seqs, batch_size):
            end = min(i + batch_size, num_seqs)
            curr_batch = msa_tensor[i:end]
            matches = (curr_batch.unsqueeze(1) == msa_tensor.unsqueeze(0)).sum(dim=2)
            m_a = (matches > threshold_match).sum(dim=1).float()
            weights[i:end] = 1.0 / m_a

        meff = weights.sum().item()
        log_meff = np.log(meff)

        del msa_tensor, weights
        torch.cuda.empty_cache()
        return meff, log_meff
    except Exception as e:
        print(f"\nError processing {a3m_path}: {e}")
        return None, None


def batch_process_meff_gpu(input_dir, identity_threshold=0.8, batch_size=2):
    if not torch.cuda.is_available():
        print("Fatal: no GPU available")
        return

    a3m_files = glob.glob(os.path.join(input_dir, "*.a3m"))
    if not a3m_files:
        print(f"No .a3m files in {input_dir}")
        return

    print(f"Found {len(a3m_files)} files. Computing log Meff (base e) on GPU...")
    print(f"{'filename':<30} | {'Meff':<12} | {'log Meff':<10}")
    print("-" * 60)

    all_log_meffs = []
    start = time.time()
    for file_path in sorted(a3m_files):
        file_name = os.path.basename(file_path)
        fstart = time.time()
        meff, log_meff = calculate_meff_gpu(file_path, identity_threshold, batch_size)
        if meff is not None:
            all_log_meffs.append(log_meff)
            print(f"{file_name:<30} | {meff:<12.2f} | {log_meff:<10.4f} ({time.time() - fstart:.1f}s)")

    if all_log_meffs:
        print("-" * 60)
        print(f"Total time: {time.time() - start:.2f} s")
        print(f"Average log Meff (base e): {np.mean(all_log_meffs):.4f}")
    else:
        print("No results.")


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated log Meff over A3M directory")
    parser.add_argument("--input_dir", required=True, help="Directory of one-line A3M files")
    parser.add_argument("--identity_threshold", type=float, default=0.8,
                        help="Pairwise identity threshold theta (default: 0.8)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Sequences per GPU batch (default: 2; 2048 ~ 8-10GB VRAM)")
    args = parser.parse_args()
    batch_process_meff_gpu(args.input_dir, args.identity_threshold, args.batch_size)


if __name__ == "__main__":
    main()
