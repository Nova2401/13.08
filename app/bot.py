from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from .config import Config
from .indexer import SimpleIndex
from .rag import RagEngine
from .scraper import scrape_all
from .embeddings import OpenAIEmbedder


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	await update.message.reply_text("Привет! Задайте вопрос по кейсам EORA.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	rag: RagEngine = context.application.bot_data["rag"]
	query = update.message.text.strip()
	try:
		answer = await asyncio.to_thread(rag.generate, query)
		await update.message.reply_text(answer, disable_web_page_preview=False)
	except Exception as e:
		await update.message.reply_text(f"Ошибка: {e}")


def ensure_index(cfg: Config) -> None:
	index = SimpleIndex(cfg)
	try:
		index.load()
		return
	except FileNotFoundError:
		pages = scrape_all(cfg)
		pairs = [(p.url, p.text) for p in pages]
		embedder = OpenAIEmbedder(cfg)
		index.build(pairs, embedder.embed_texts)
		return


def main() -> None:
	load_dotenv()
	cfg = Config.load()
	if not cfg.telegram_bot_token:
		raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
	ensure_index(cfg)
	index = SimpleIndex(cfg)
	index.load()
	rag = RagEngine(cfg, index)

	app = ApplicationBuilder().token(cfg.telegram_bot_token).build()
	app.bot_data["rag"] = rag

	app.add_handler(CommandHandler("start", handle_start))
	app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

	app.run_polling()


if __name__ == "__main__":
	main()
