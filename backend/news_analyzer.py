"""
Módulo de análisis de noticias y sentiment.

Objetivo: reunir titulares positivos y negativos (geopolítica, energía,
innovación, IA, macroeconomía) y convertirlos en un puntaje [-1, +1].
"""

from __future__ import annotations

import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal

import httpx
from dotenv import load_dotenv

SentimentScore = float

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # NewsAPI.org u otra fuente similar


@dataclass
class Topic:
    name: str
    query: str
    weight: float
    polarity: Literal["positive", "negative", "neutral"]


DEFAULT_TOPICS: list[Topic] = [
    Topic(name="Geopolítica", query="war OR conflict OR tensions", weight=1.2, polarity="negative"),
    Topic(name="Energía/Recursos", query="oil discovery OR lithium found", weight=0.9, polarity="positive"),
    Topic(name="IA/Innovación", query="artificial intelligence breakthrough", weight=1.0, polarity="positive"),
    Topic(name="Regulación", query="crypto regulation crackdown", weight=1.1, polarity="negative"),
    Topic(name="Macroeconomía", query="central bank interest rates", weight=1.0, polarity="neutral"),
]


async def _fetch_news(topic: Topic, page_size: int = 10) -> list[str]:
    """
    Recupera titulares usando NewsAPI (o genera dummy si falta API key).
    """
    if not NEWS_API_KEY:
        # Fallback rápido para desarrollo sin API real
        seed = "positivo" if topic.polarity == "positive" else "negativo"
        return [f"{topic.name}: titular {seed} {i}" for i in range(1, 4)]

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic.query,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("articles", [])
        return [article["title"] for article in articles if article.get("title")]


def _score_headlines(headlines: Iterable[str], topic: Topic) -> float:
    """
    Asigna un puntaje heurístico según palabras clave positivas/negativas.
    """
    if not headlines:
        return 0.0

    positive_cues = ("growth", "up", "record", "profit", "breakthrough", "discovery")
    negative_cues = ("war", "conflict", "ban", "crash", "sanction", "hack", "scam")

    counters = Counter()
    score = 0.0

    for title in headlines:
        text = title.lower()
        if any(word in text for word in positive_cues):
            counters["positive"] += 1
            score += 0.3
        if any(word in text for word in negative_cues):
            counters["negative"] += 1
            score -= 0.4

    # Ajusta por el sesgo del tópico (ej. geopolítica tiende a ser negativa)
    if topic.polarity == "positive":
        score += 0.1 * len(headlines)
    elif topic.polarity == "negative":
        score -= 0.1 * len(headlines)

    # Se pondera por el peso del tópico
    return score * topic.weight


async def analyze_sentiment(
    asset: str,
    mode: Literal["llm", "rules"] = "llm",
    topics: list[Topic] | None = None,
) -> SentimentScore:
    """
    Devuelve un puntaje entre -1 y +1.

    Parameters
    ----------
    asset:
        Símbolo del activo (BTC/USDT, EUR/USD, etc.).
    mode:
        - "llm": (futuro) delega el análisis en un modelo de lenguaje.
        - "rules": aplica heurísticas rápidas (actual).
    topics:
        Lista personalizada de tópicos/consultas.
    """

    topics = topics or DEFAULT_TOPICS

    if mode == "llm":
        # Por ahora reutilizamos la heurística, pero dejando el hook listo
        mode = "rules"

    aggregated_score = 0.0

    for topic in topics:
        headlines = await _fetch_news(topic)
        aggregated_score += _score_headlines(headlines, topic)

    # Normalizamos a rango [-1, +1]
    normalized = max(-1.0, min(1.0, aggregated_score / 20))

    # Pequeña variación para evitar empates perfectos
    noise = random.uniform(-0.05, 0.05)
    final_score = max(-1.0, min(1.0, normalized + noise))

    return final_score


