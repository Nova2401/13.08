from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
	project_root: Path
	data_dir: Path
	sources_file: Path
	openai_api_key: str
	openai_chat_model: str
	openai_embed_model: str
	chunk_size: int
	chunk_overlap: int
	top_k: int
	request_timeout: int
	telegram_bot_token: str | None

	@staticmethod
	def load() -> "Config":
		project_root = Path(__file__).resolve().parents[1]
		data_dir = project_root / "data"
		sources_file = data_dir / "sources.txt"
		openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
		openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
		openai_embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()
		chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
		chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
		top_k = int(os.getenv("TOP_K", "6"))
		request_timeout = int(os.getenv("REQUEST_TIMEOUT", "20"))
		telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

		return Config(
			project_root=project_root,
			data_dir=data_dir,
			sources_file=sources_file,
			openai_api_key=openai_api_key,
			openai_chat_model=openai_chat_model,
			openai_embed_model=openai_embed_model,
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
			top_k=top_k,
			request_timeout=request_timeout,
			telegram_bot_token=telegram_bot_token,
		)
