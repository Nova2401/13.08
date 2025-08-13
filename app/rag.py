from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from openai import OpenAI
from .config import Config
from .indexer import SimpleIndex


SYSTEM_INSTRUCTION = (
	"""
	Ты опытный ассистент компании EORA. Отвечай по-русски кратко и по делу, строго опираясь на предоставленный контекст.
	Если контекст недостаточен — явно скажи об этом. В конце ответа всегда перечисли использованные источники в формате:\nИсточники: [<номер>] <ссылка>, ...
	"""
).strip()


@dataclass
class RetrievedChunk:
	text: str
	source_url: str
	order: int


class RagEngine:
	def __init__(self, cfg: Config, index: SimpleIndex) -> None:
		self.cfg = cfg
		self.index = index
		self.client = OpenAI(api_key=cfg.openai_api_key)
		if not cfg.openai_api_key:
			raise RuntimeError("OPENAI_API_KEY is not set")

	def embed_query(self, query: str) -> np.ndarray:
		resp = self.client.embeddings.create(model=self.cfg.openai_embed_model, input=[query])
		vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
		return vec

	def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
		if self.index.embeddings is None or self.index.meta is None:
			self.index.load()
		qv = self.embed_query(query)
		idxs = self.index.search(qv, top_k or self.cfg.top_k)
		ctx = self.index.get_context(idxs)
		return [RetrievedChunk(text=c["text"], source_url=c["source_url"], order=c["order"]) for c in ctx]

	def build_prompt(self, query: str, chunks: List[RetrievedChunk]) -> Tuple[str, List[str]]:
		unique_links: List[str] = []
		link_to_id: dict[str, int] = {}
		context_parts: List[str] = []
		for chunk in chunks:
			if chunk.source_url not in link_to_id:
				link_to_id[chunk.source_url] = len(unique_links) + 1
				unique_links.append(chunk.source_url)
			context_parts.append(f"[Источник {link_to_id[chunk.source_url]}] {chunk.source_url}\n{chunk.text}")
		context = "\n\n".join(context_parts)
		user_prompt = (
			"Контекст:\n" + context + "\n\n" +
			"Вопрос: " + query + "\n" +
			"Ответ:"
		)
		return user_prompt, unique_links

	def generate(self, query: str) -> str:
		chunks = self.retrieve(query)
		user_prompt, links = self.build_prompt(query, chunks)
		resp = self.client.chat.completions.create(
			model=self.cfg.openai_chat_model,
			messages=[
				{"role": "system", "content": SYSTEM_INSTRUCTION},
				{"role": "user", "content": user_prompt},
			],
			temperature=0.2,
		)
		answer = resp.choices[0].message.content.strip()
		sources_list = ", ".join([f"[{i+1}] {link}" for i, link in enumerate(links)])
		if "Источники:" not in answer:
			answer += f"\n\nИсточники: {sources_list}"
		return answer
