"""ESMC-600M embedding generation for query sequences (BWMSA online phase)."""

import numpy as np
import torch
import faiss
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_esmc(device=None):
    device = device or _device()
    return ESMC.from_pretrained("esmc_600m").to(device)


def embed_sequence(client, sequence):
    """Run ESMC on one sequence and return its BOS/EOS-trimmed mean-pooled embedding.

    The ESMC tokenizer adds BOS at index 0 and EOS at index -1, so logits.embeddings
    has shape [1, L+2, dim] for a length-L input; we slice [:, 1:-1, :] before averaging."""
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    out = client.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True),
    )
    pooled = out.embeddings[:, 1:-1, :].mean(dim=1)
    return pooled.cpu().numpy().squeeze()


class ProteinEmbeddingGenerator:
    """Query-time embedder: loads whitening params and produces whitened, L2-normalized vectors."""

    def __init__(self, whiten_params_path):
        print("Initializing protein embedding generator...")
        print("Loading whitening parameters...")
        params = np.load(whiten_params_path)
        self.kernel = params["kernel"]
        self.bias = params["bias"]
        self.col_mean = params["col_mean"]
        print(
            f"Whitening parameters loaded: kernel shape={self.kernel.shape}, "
            f"bias shape={self.bias.shape}"
        )

        print("Loading ESMC model...")
        self.client = load_esmc()
        print("ESMC model loaded")

    def _apply_whitening(self, embedding):
        embedding = embedding.astype(np.float64)
        embedding = np.where(np.isnan(embedding), self.col_mean, embedding)
        whitened = (embedding.reshape(1, -1) + self.bias).dot(self.kernel).astype(np.float32)
        faiss.normalize_L2(whitened)
        return whitened.squeeze()

    def generate_embeddings(self, fasta_file):
        print(f"Generating embeddings from {fasta_file}...")
        embeddings = []
        identifiers = []
        for num, record in enumerate(SeqIO.parse(fasta_file, "fasta"), start=1):
            seq_id = record.id
            print(f"Processing sequence {num}: {seq_id}")
            raw = embed_sequence(self.client, str(record.seq))
            whitened = self._apply_whitening(raw)
            embeddings.append(whitened)
            identifiers.append(f"_{num}_{seq_id}")
            print(f"Generated whitened embedding, shape: {whitened.shape}")

        print(f"Completed! Processed {len(embeddings)} sequences in total")
        return embeddings, identifiers
