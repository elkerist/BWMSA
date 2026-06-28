"""Build a combined MSA by stitching BWMSA's A3M with ColabFold's UniRef + environmental A3Ms.

Steps per target id <i> found in --iter3_dir:
  1. Concatenate ColabFold's uniref.a3m with BWMSA's <i>.a3m (skipping the query header).
  2. Run hhfilter -id 90 -diff 3000 over the merged file.
  3. Append ColabFold's environmental A3M (bfd.mgnify30.metaeuk30.smag30.a3m) to the result.

ColabFold layout expected:
  <colab_dir>/results<i>/<some_subdir>/uniref.a3m
  <colab_dir>/results<i>/<some_subdir>/bfd.mgnify30.metaeuk30.smag30.a3m
"""

import argparse
import glob
import os
import subprocess


def combine_one(target_id, iter3_dir, colab_dir, temp_dir, final_dir):
    iter3_file = os.path.join(iter3_dir, f"{target_id}.a3m")
    results_subdir = os.path.join(colab_dir, f"results{target_id}")

    if not os.path.exists(iter3_file) or not os.path.exists(results_subdir):
        print(f"Skipping {target_id}: file or directory missing")
        return False

    subfolders = [f.path for f in os.scandir(results_subdir) if f.is_dir()]
    if not subfolders:
        return False

    target_subfolder = subfolders[0]
    uniref_path = os.path.join(target_subfolder, "uniref.a3m")
    env_path = os.path.join(target_subfolder, "bfd.mgnify30.metaeuk30.smag30.a3m")

    print(f"[{target_id}] first concatenation (uniref + iter3 body)...")
    with open(uniref_path, "r") as f:
        uniref_content = f.read()
    with open(iter3_file, "r") as f:
        iter3_lines = f.readlines()

    # Skip the BWMSA file's first record (the query itself) — find the 2nd header.
    second_seq_idx = -1
    header_count = 0
    for idx, line in enumerate(iter3_lines):
        if line.startswith(">"):
            header_count += 1
            if header_count == 2:
                second_seq_idx = idx
                break

    remaining_iter3 = "".join(iter3_lines[second_seq_idx:]) if second_seq_idx != -1 else ""

    combined_temp_path = os.path.join(temp_dir, f"{target_id}.a3m")
    with open(combined_temp_path, "w") as f:
        f.write(uniref_content)
        if not uniref_content.endswith("\n"):
            f.write("\n")
        f.write(remaining_iter3)

    filtered_path = os.path.join(temp_dir, f"{target_id}_filtered.a3m")
    print(f"[{target_id}] running hhfilter...")
    cmd = ["hhfilter", "-i", combined_temp_path, "-o", filtered_path, "-id", "90", "-diff", "3000"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: hhfilter failed for {target_id}: {e}")
        return False

    print(f"[{target_id}] final concatenation (filtered + env)...")
    with open(filtered_path, "r") as f:
        filtered_content = f.read()
    with open(env_path, "r") as f:
        env_content = f.read()

    final_file_path = os.path.join(final_dir, f"{target_id}.a3m")
    with open(final_file_path, "w") as f:
        f.write(filtered_content)
        if not filtered_content.endswith("\n"):
            f.write("\n")
        f.write(env_content)
    return True


def main():
    parser = argparse.ArgumentParser(description="Build combined MSA from BWMSA + ColabFold A3Ms")
    parser.add_argument("--iter3_dir", required=True, help="Directory of BWMSA A3M files (<id>.a3m)")
    parser.add_argument("--colab_dir", required=True, help="Directory of ColabFold per-target results subdirs")
    parser.add_argument("--temp_dir", required=True, help="Directory for intermediate merged/filtered files")
    parser.add_argument("--final_dir", required=True, help="Directory for final combined A3M output")
    args = parser.parse_args()

    for d in [args.temp_dir, args.final_dir]:
        os.makedirs(d, exist_ok=True)

    a3m_paths = sorted(glob.glob(os.path.join(args.iter3_dir, "*.a3m")))
    target_ids = [os.path.splitext(os.path.basename(p))[0] for p in a3m_paths]

    if not target_ids:
        print(f"No .a3m files found in {args.iter3_dir}")
        return

    print(f"Found {len(target_ids)} targets to combine")
    for target_id in target_ids:
        combine_one(target_id, args.iter3_dir, args.colab_dir, args.temp_dir, args.final_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
