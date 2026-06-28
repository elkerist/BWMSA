"""Online FAISS retrieval against the whitened UniRef90 index."""

import mmap
import os
import time

import faiss
import numpy as np


class OptimizedSearcher:
    """Loads the FAISS index + builds an mmap-backed FASTA position index for fast retrieval."""

    def __init__(self, trained_index, fasta_path, index_path):
        print("Loading FAISS index...")
        t0 = time.time()
        self.index = faiss.read_index(trained_index)
        print(f"FAISS index loaded, time taken: {time.time() - t0:.2f} seconds")

        print("Loading sequence index...")
        t0 = time.time()

        # We store the position index as a compressed npz; if a legacy .json
        # path was passed in, redirect it to its .npz sibling.
        numpy_index_path = index_path.replace(".json", ".npz")

        if not os.path.exists(numpy_index_path):
            print("Sequence index does not exist, creating...")
            self._create_sequence_index(fasta_path, numpy_index_path)

        index_data = np.load(numpy_index_path)
        self.seq_starts = index_data["starts"]
        self.seq_ends = index_data["ends"]

        self.fasta_file = open(fasta_path, "r")
        self.fasta_mmap = mmap.mmap(self.fasta_file.fileno(), 0, access=mmap.ACCESS_READ)

        print(
            f"Sequence index loaded, {len(self.seq_starts)} sequences, "
            f"time taken: {time.time() - t0:.2f} seconds"
        )

    def _create_sequence_index(self, fasta_path, index_path):
        """Scan the FASTA once to record byte offsets of each record."""
        print("Analyzing FASTA file to create index...")
        starts = []
        ends = []

        with open(fasta_path, "rb") as f:
            current_seq_start = None
            line_count = 0

            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break

                line_str = line.decode("utf-8").strip()
                line_count += 1
                if line_count % 1_000_000 == 0:
                    print(f"Processed {line_count} lines...")

                if line_str.startswith(">"):
                    if current_seq_start is not None:
                        starts.append(current_seq_start)
                        ends.append(pos)
                    current_seq_start = pos

            if current_seq_start is not None:
                starts.append(current_seq_start)
                ends.append(f.tell())

        starts_array = np.array(starts, dtype=np.int64)
        ends_array = np.array(ends, dtype=np.int64)

        print(f"Saving sequence index to {index_path}...")
        np.savez_compressed(index_path, starts=starts_array, ends=ends_array)
        print(f"Sequence index created, {len(starts_array)} sequences")

    def get_sequence_by_index(self, idx):
        if idx >= len(self.seq_starts):
            raise IndexError(f"Sequence index {idx} out of range")

        start_pos = self.seq_starts[idx]
        end_pos = self.seq_ends[idx]

        self.fasta_mmap.seek(start_pos)
        length = end_pos - start_pos
        return self.fasta_mmap.read(length).decode("utf-8").strip()

    def get_sequences_batch(self, indices):
        """Fetch sequences in sorted-by-offset order, then restore caller order."""
        sorted_indices = sorted(
            enumerate(indices), key=lambda x: self.seq_starts[x[1]]
        )

        results = [None] * len(indices)
        for original_pos, idx in sorted_indices:
            results[original_pos] = self.get_sequence_by_index(idx)
        return results

    def search_batch(self, batch_embeddings, topk=100_000):
        """Stack and run a single FAISS query for all rows at once."""
        print(f"Executing batch search, batch size: {len(batch_embeddings)}")
        t0 = time.time()

        batch_matrix = np.vstack(batch_embeddings)
        distances, indices = self.index.search(batch_matrix, topk)
        print(f"FAISS search completed, time taken: {time.time() - t0:.2f} seconds")
        return distances, indices

    def save_results_with_sequences(self, indices_list, identifiers, output_dir):
        """For each query, fetch the topk candidate records and write to candidates{id}.fasta."""
        print("Getting sequences by indices and saving results...")
        t0 = time.time()

        for identifier, indices in zip(identifiers, indices_list):
            print(f"Processing query {identifier}, getting {len(indices)} sequences...")
            seq_t0 = time.time()
            sequences = self.get_sequences_batch(indices)
            print(f"Sequence retrieval completed, time taken: {time.time() - seq_t0:.2f} seconds")

            output_path = os.path.join(output_dir, f"candidates{identifier}.fasta")
            with open(output_path, "w") as output_handle:
                for seq_data in sequences:
                    output_handle.write(seq_data + "\n")

            print(f"Saved to {output_path}")

        print(f"All results saved, time taken: {time.time() - t0:.2f} seconds")

    def __del__(self):
        if hasattr(self, "fasta_mmap"):
            self.fasta_mmap.close()
        if hasattr(self, "fasta_file"):
            self.fasta_file.close()
