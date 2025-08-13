from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np
from openai import OpenAI
from .config import Config


@dataclass
class EmbeddingStore:
	embeddings_path: Path
	index_meta_path: Path

	def save(self, embeddings: np.ndarray, meta: dict) -> None:
		self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
		np.save(self.embeddings_path, embeddings)
		self.index_meta_path.write_text(__import__("json").dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

	def load(self) -> tuple[np.ndarray, dict]:
		emb = np.load(self.embeddings_path)
		meta = __import__("json").loads(self.index_meta_path.read_text(encoding="utf-8"))
		return emb, meta


class OpenAIEmbedder:
	def __init__(self, cfg: Config) -> None:
		self.cfg = cfg
		self.client = OpenAI(api_key=cfg.openai_api_key)
		if not cfg.openai_api_key:
			raise RuntimeError("OPENAI_API_KEY is not set")

	def embed_texts(self, texts: List[str]) -> np.ndarray:
		resp = self.client.embeddings.create(model=self.cfg.openai_embed_model, input=texts)
		vectors = [item.embedding for item in resp.data]
		return np.asarray(vectors, dtype=np.float32)
