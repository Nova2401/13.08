from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from .config import Config
from .scraper import scrape_all
from .embeddings import OpenAIEmbedder
from .indexer import SimpleIndex


def main() -> None:
	load_dotenv()
	cfg = Config.load()
	cfg.data_dir.mkdir(parents=True, exist_ok=True)
	pages = scrape_all(cfg, verbose=True)
	if not pages:
		print("Нет доступных страниц для индексации (все упали или пустые). Проверьте сеть/SSL.")
		return
	pairs = [(p.url, p.text) for p in pages]

	embedder = OpenAIEmbedder(cfg)
	index = SimpleIndex(cfg)
	index.build(pairs, embedder.embed_texts)
	print(f"Indexed {len(pairs)} pages into {index.embeddings.shape[0]} chunks")


if __name__ == "__main__":
	main()
