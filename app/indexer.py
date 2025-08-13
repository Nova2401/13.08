from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from .config import Config
from .utils import Chunk, split_text_into_chunks, cosine_similarity_matrix


@dataclass
class DocumentChunk:
	id: int
	source_url: str
	text: str
	order: int


class SimpleIndex:
	def __init__(self, cfg: Config, embeddings_path: Path | None = None, meta_path: Path | None = None) -> None:
		self.cfg = cfg
		self.embeddings_path = embeddings_path or (cfg.data_dir / "embeddings.npy")
		self.meta_path = meta_path or (cfg.data_dir / "index.json")
		self.embeddings: np.ndarray | None = None
		self.meta: Dict[str, Any] | None = None

	def build(self, pages: List[tuple[str, str]], embed_fn) -> None:
		chunks: List[DocumentChunk] = []
		texts: List[str] = []
		chunk_id = 0
		for url, text in pages:
			parts = split_text_into_chunks(text, self.cfg.chunk_size, self.cfg.chunk_overlap)
			for i, part in enumerate(parts):
				chunks.append(DocumentChunk(id=chunk_id, source_url=url, text=part, order=i))
				texts.append(part)
				chunk_id += 1
		if not texts:
			raise RuntimeError("No text chunks to index")
		vectors: np.ndarray = embed_fn(texts)
		self.embeddings = vectors
		self.meta = {
			"chunks": [{"id": c.id, "source_url": c.source_url, "order": c.order} for c in chunks],
			"texts": texts,
		}
		self.save()

	def save(self) -> None:
		assert self.embeddings is not None and self.meta is not None
		self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
		np.save(self.embeddings_path, self.embeddings)
		self.meta_path.write_text(__import__("json").dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

	def load(self) -> None:
		self.embeddings = np.load(self.embeddings_path)
		self.meta = __import__("json").loads(self.meta_path.read_text(encoding="utf-8"))

	def search(self, query_vector: np.ndarray, top_k: int) -> List[int]:
		assert self.embeddings is not None
		S = cosine_similarity_matrix(self.embeddings, query_vector[np.newaxis, :])
		scores = S[:, 0]
		k = min(top_k, scores.shape[0])
		idxs = np.argpartition(-scores, k - 1)[:k]
		idxs = idxs[np.argsort(-scores[idxs])]
		return idxs.tolist()

	def get_context(self, indices: List[int]) -> List[dict]:
		assert self.meta is not None
		texts: List[str] = self.meta["texts"]
		chunks_meta = self.meta["chunks"]
		return [
			{
				"text": texts[i],
				"source_url": chunks_meta[i]["source_url"],
				"order": chunks_meta[i]["order"],
			}
			for i in indices
		]
