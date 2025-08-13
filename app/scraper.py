from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import Config
from .utils import http_get, extract_main_text


@dataclass
class Page:
	url: str
	text: str


def read_sources_file(path: Path) -> List[str]:
	lines: List[str] = []
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			lines.append(line)
	return lines


def scrape_all(cfg: Config, verbose: bool = True) -> List[Page]:
	urls = read_sources_file(cfg.sources_file)
	pages: List[Page] = []
	for url in urls:
		try:
			html = http_get(url, timeout=cfg.request_timeout)
			text = extract_main_text(html)
			if text:
				pages.append(Page(url=url, text=text))
				if verbose:
					print(f"OK: {url}")
		except Exception as e:
			if verbose:
				print(f"SKIP ({type(e).__name__}): {url} -> {e}")
	return pages
