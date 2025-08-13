from __future__ import annotations
import sys
from dotenv import load_dotenv
from .config import Config
from .indexer import SimpleIndex
from .rag import RagEngine


def main() -> None:
	load_dotenv()
	if len(sys.argv) < 2:
		print("Usage: python -m app.cli \"<ваш вопрос>\"")
		sys.exit(1)
	query = sys.argv[1]
	cfg = Config.load()
	index = SimpleIndex(cfg)
	index.load()
	rag = RagEngine(cfg, index)
	answer = rag.generate(query)
	print(answer)


if __name__ == "__main__":
	main()
