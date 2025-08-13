from __future__ import annotations

import os
import re
from dataclasses import dataclass
from html import unescape
from typing import Iterable, List, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _build_session() -> requests.Session:
	s = requests.Session()
	retry = Retry(
		total=5,
		connect=5,
		read=5,
		backoff_factor=0.5,
		status_forcelist=[429, 500, 502, 503, 504],
		respect_retry_after_header=True,
	)
	adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
	s.mount("http://", adapter)
	s.mount("https://", adapter)
	return s


def http_get(url: str, timeout: int = 20) -> str:
	session = _build_session()
	headers = {
		"User-Agent": (
			"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
			"AppleWebKit/537.36 (KHTML, like Gecko) "
			"Chrome/126.0.0.0 Safari/537.36"
		),
		"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
		"Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
		"Connection": "keep-alive",
	}
	try:
		resp = session.get(url, timeout=timeout, headers=headers)
		resp.raise_for_status()
		return resp.text
	except requests.exceptions.SSLError:
		if os.getenv("RAG_INSECURE", "0") == "1":
			resp = session.get(url, timeout=timeout, headers=headers, verify=False)
			resp.raise_for_status()
			return resp.text
		raise


def extract_main_text(html: str) -> str:
	soup = BeautifulSoup(html, "lxml")
	main = soup.find("main") or soup
	for tag in main.find_all(["script", "style", "nav", "footer", "form", "noscript"]):
		tag.decompose()
	text = main.get_text("\n")
	text = unescape(text)
	text = re.sub(r"\n{2,}", "\n\n", text)
	text = re.sub(r"\s+", " ", text)
	text = text.strip()
	return text


@dataclass
class Chunk:
	source_url: str
	text: str
	order: int


def split_text_into_chunks(text: str, max_size: int, overlap: int) -> List[str]:
	if max_size <= 0:
		return [text]
	chunks: List[str] = []
	start = 0
	length = len(text)
	while start < length:
		end = min(start + max_size, length)
		chunks.append(text[start:end])
		if end == length:
			break
		start = max(0, end - overlap)
	return chunks


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
	b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
	return np.matmul(a_norm, b_norm.T)


def top_k_indices(similarities: np.ndarray, k: int) -> List[int]:
	k = min(k, similarities.shape[0])
	return np.argpartition(-similarities, k - 1, axis=0)[:k].flatten().tolist()
