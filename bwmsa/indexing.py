"""Incremental whitening parameter estimation and FAISS index build (offline phase).

The database embeddings are expected as a pre-computed .npy on disk
(shape [N, D], typically D=1152 for ESMC-600M). Everything here streams
via mmap so the full matrix never has to fit in RAM.
"""

import gc
import faiss
import numpy as np


def compute_kernel_bias(emb_file, n_components=512, batch_size=1_000_000):
    """Two-pass incremental whitening parameters from an .npy on disk.

    Pass 1: column-wise mean ignoring NaN (np.nansum + valid count).
    Pass 2: covariance Σ = Σ (x - µ)ᵀ(x - µ) / (N - 1) after NaN-fill with column mean.
    Then SVD(Σ) = U Λ Uᵀ and W = U Λ^(-1/2). The returned bias is -µ so callers
    can write (x + bias) · W."""
    embeddings = np.load(emb_file, mmap_mode="r")
    N, D = embeddings.shape

    print("  computing column mean (incremental, NaN-aware)...")
    sum_vec = np.zeros(D, dtype=np.float64)
    count_vec = np.zeros(D, dtype=np.int64)

    for i in range(0, N, batch_size):
        batch = embeddings[i : i + batch_size]
        sum_vec += np.nansum(batch, axis=0)
        count_vec += (~np.isnan(batch)).sum(axis=0)

        if i % (batch_size * 10) == 0:
            print(f"    mean progress: {i / N * 100:.1f}%")

    col_mean = (sum_vec / count_vec).astype(np.float32)
    mu = col_mean.reshape(1, -1)

    print("  computing covariance matrix (incremental)...")
    cov_sum = np.zeros((D, D), dtype=np.float64)
    total_count = 0

    for i in range(0, N, batch_size):
        batch = embeddings[i : i + batch_size].astype(np.float32)
        batch_clean = np.where(np.isnan(batch), col_mean, batch)
        batch_centered = batch_clean - mu
        cov_sum += batch_centered.T @ batch_centered
        total_count += batch_clean.shape[0]

        del batch, batch_clean, batch_centered
        if i % (batch_size * 10) == 0:
            print(f"    covariance progress: {i / N * 100:.1f}%")

    cov = (cov_sum / (total_count - 1)).astype(np.float32)

    print("  SVD decomposition...")
    u, s, _vh = np.linalg.svd(cov)

    print("  building whitening matrix...")
    W = np.dot(u, np.diag(1 / np.sqrt(s))).astype(np.float32)

    return W[:, :n_components], -mu, col_mean


def build_index(
    emb_file,
    index_out,
    params_out,
    n_components=512,
    batch_size=1_000_000,
):
    """Build a whitened FAISS IndexFlatIP from a pre-computed embedding .npy."""
    print("loading embedding metadata...")
    embeddings = np.load(emb_file, mmap_mode="r")
    total_rows, dim = embeddings.shape
    print(f"total rows: {total_rows}, dim: {dim}")

    print("computing whitening parameters...")
    kernel, bias, col_mean = compute_kernel_bias(
        emb_file, n_components=n_components, batch_size=batch_size
    )

    dim_whitened = kernel.shape[1]
    print(f"creating IndexFlatIP({dim_whitened})...")
    index = faiss.IndexFlatIP(dim_whitened)

    print("streaming batches: NaN-fill, whiten, L2-normalize, add to index...")
    for i in range(0, total_rows, batch_size):
        batch = embeddings[i : i + batch_size].astype(np.float32)
        batch = np.where(np.isnan(batch), col_mean, batch)

        whitened = (batch + bias).dot(kernel)
        faiss.normalize_L2(whitened)
        index.add(whitened)

        del batch, whitened
        gc.collect()

        if i % (batch_size * 10) == 0:
            print(f"  progress: {i / total_rows * 100:.1f}% - added {index.ntotal:,}")

    print(f"saving index to {index_out}...")
    faiss.write_index(index, index_out)
    print(f"saving whitening params to {params_out}...")
    np.savez(params_out, kernel=kernel, bias=bias, col_mean=col_mean)

    print(f"done! index size: {index.ntotal:,}, whitened dim: {dim_whitened}")
