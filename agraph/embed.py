"""SPECTER2 encoder (allenai/specter2_base + task adapters)."""
from __future__ import annotations

import numpy as np
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

# adapters logs "none are activated" while loading, before set_active=True runs.
hf_logging.set_verbosity_error()

from .config import SPECTER_BASE, SPECTER_DOC_ADAPTER, SPECTER_QUERY_ADAPTER


class Specter2:
    """mode='doc' for papers (title [SEP] abstract), mode='query' for short search text."""

    def __init__(self, mode: str = "doc", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(SPECTER_BASE)
        self.model = AutoAdapterModel.from_pretrained(SPECTER_BASE)
        adapter = SPECTER_DOC_ADAPTER if mode == "doc" else SPECTER_QUERY_ADAPTER
        self.model.load_adapter(adapter, source="hf", load_as=mode, set_active=True)
        self.model.to(self.device).eval()

    def paper_text(self, title: str, abstract: str | None) -> str:
        return title + (self.tok.sep_token + abstract if abstract else "")

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 64, max_length: int = 512,
               progress: bool = False) -> np.ndarray:
        """Returns L2-normalised float32 vectors (CLS pooling), in input order."""
        order = np.argsort([-len(t) for t in texts])  # length-sorted batches = less padding
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        it = range(0, len(texts), batch_size)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc="embedding", unit="batch")
        for i in it:
            idx = order[i:i + batch_size]
            enc = self.tok([texts[j] for j in idx], padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt",
                           return_token_type_ids=False).to(self.device)
            with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                h = self.model(**enc).last_hidden_state[:, 0, :]
            h = torch.nn.functional.normalize(h.float(), dim=-1)
            out[idx] = h.cpu().numpy()
        return out
