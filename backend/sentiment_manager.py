"""
Sentiment Manager - Gestor central del sistema de análisis de sentimiento.

Coordina el news fetcher y sentiment analyzer, mantiene histórico,
y proporciona agregados de sentimiento del mercado.
"""

import asyncio
import time
from typing import Dict, List, Optional
from datetime import datetime, timezone
from pathlib import Path
import json

from news_fetcher import NewsFetcher
from sentiment_analyzer import SentimentAnalyzer

class SentimentManager:
    """Gestor central del sistema de sentimiento."""
    
    def __init__(self):
        self.fetcher = NewsFetcher()
        self.analyzer = SentimentAnalyzer()
        self.history_file = Path(__file__).parent / ".cache" / "sentiment_history.json"
        self.update_interval = 900  # 15 minutos
        self.last_update = 0
        self.current_sentiment = None
        self.news_history = []
        
        # Intentar cargar histórico
        self._load_history()
    
    async def update(self):
        """Actualiza el sentimiento del mercado (llamar periódicamente)."""
        # Verificar si necesitamos actualizar
        if time.time() - self.last_update < self.update_interval:
            return self.current_sentiment
        
        print("[SENTIMENT] 🔄 Actualizando análisis de sentimiento...")
        
        try:
            # 1. Fetch noticias
            raw_news = await self.fetcher.fetch_latest_news(limit=15)
            
            # 2. Analizar sentimiento de cada noticia
            analyzed_news = []
            for news_item in raw_news:
                analyzed = self.analyzer.analyze_news_item(news_item.copy())
                analyzed_news.append(analyzed)
            
            # 3. Calcular sentimiento agregado
            self.current_sentiment = self._calculate_aggregate_sentiment(analyzed_news)
            
            # 4. Guardar en histórico
            self.news_history = analyzed_news[:10]  # Últimas 10 noticias
            self._save_history()
            
            self.last_update = time.time()
            
            # Log resumen
            overall = self.current_sentiment["overall"]
            score = self.current_sentiment["score"]
            emoji = {"bullish": "🐂", "bearish": "🐻", "neutral": "😐"}.get(overall, "😐")
            print(f"[SENTIMENT] {emoji} Mercado {overall.upper()} (score: {score:.2f})")
            
        except Exception as e:
            print(f"[SENTIMENT] ❌ Error actualizando: {e}")
            # Mantener sentimiento anterior o neutral por defecto
            if not self.current_sentiment:
                self.current_sentiment = self._get_neutral_sentiment()
        
        return self.current_sentiment
    
    def get_current_sentiment(self) -> Dict:
        """
        Retorna el sentimiento actual del mercado.
        
        Returns:
            {
                "overall": "bullish" | "bearish" | "neutral",
                "score": float (-1.0 a +1.0),
                "confidence": "high" | "medium" | "low",
                "recent_events": [...],  # Últimas 3 noticias más relevantes
                "trend": "improving" | "worsening" | "stable",
                "last_updated": timestamp
            }
        """
        if not self.current_sentiment:
            return self._get_neutral_sentiment()
        
        return self.current_sentiment
    
    def get_recent_news(self, limit: int = 10) -> List[Dict]:
        """Retorna las últimas noticias analizadas."""
        return self.news_history[:limit]
    
    def _calculate_aggregate_sentiment(self, news_list: List[Dict]) -> Dict:
        """Calcula el sentimiento agregado de una lista de noticias."""
        if not news_list:
            return self._get_neutral_sentiment()
        
        # Filtrar noticias válidas (con sentimiento)
        valid_news = [n for n in news_list if n.get("sentiment")]
        if not valid_news:
            return self._get_neutral_sentiment()
        
        # Calcular score ponderado (noticias de mayor impacto pesan más)
        weighted_scores = []
        impact_weights = {"high": 3.0, "medium": 2.0, "low": 1.0}
        
        for news in valid_news:
            sentiment = news["sentiment"]
            score = sentiment["score"]
            impact = sentiment.get("impact", "low")
            weight = impact_weights.get(impact, 1.0)
            
            weighted_scores.append(score * weight)
        
        # Score promedio ponderado
        if weighted_scores:
            avg_score = sum(weighted_scores) / len(weighted_scores)
        else:
            avg_score = 0.0
        
        # Determinar overall sentiment
        if avg_score > 0.25:
            overall = "bullish"
        elif avg_score < -0.25:
            overall = "bearish"
        else:
            overall = "neutral"
        
        # Calcular confianza (basado en cantidad y calidad de datos)
        high_impact_count = sum(1 for n in valid_news if n["sentiment"].get("impact") == "high")
        if len(valid_news) >= 10 and high_impact_count >= 3:
            confidence = "high"
        elif len(valid_news) >= 5:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Determinar tendencia (comparar con histórico previo)
        trend = self._calculate_trend(avg_score)
        
        # Seleccionar top 3 noticias más relevantes (por impacto y score)
        sorted_news = sorted(
            valid_news,
            key=lambda n: (
                impact_weights.get(n["sentiment"].get("impact", "low"), 1.0),
                abs(n["sentiment"]["score"])
            ),
            reverse=True
        )
        recent_events = sorted_news[:3]
        
        return {
            "overall": overall,
            "score": round(avg_score, 2),
            "confidence": confidence,
            "recent_events": recent_events,
            "trend": trend,
            "last_updated": time.time(),
            "news_count": len(valid_news)
        }
    
    def _calculate_trend(self, current_score: float) -> str:
        """Calcula si el sentimiento está mejorando o empeorando."""
        if not self.current_sentiment:
            return "stable"
        
        previous_score = self.current_sentiment.get("score", 0.0)
        delta = current_score - previous_score
        
        if delta > 0.15:
            return "improving"
        elif delta < -0.15:
            return "worsening"
        else:
            return "stable"
    
    def _get_neutral_sentiment(self) -> Dict:
        """Retorna un sentimiento neutral por defecto."""
        return {
            "overall": "neutral",
            "score": 0.0,
            "confidence": "low",
            "recent_events": [],
            "trend": "stable",
            "last_updated": time.time(),
            "news_count": 0
        }
    
    def _load_history(self):
        """Carga histórico desde archivo."""
        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text())
                self.current_sentiment = data.get("current_sentiment")
                self.news_history = data.get("news_history", [])
                self.last_update = data.get("last_update", 0)
            except:
                pass
    
    def _save_history(self):
        """Guarda histórico en archivo."""
        try:
            self.history_file.parent.mkdir(exist_ok=True)
            data = {
                "current_sentiment": self.current_sentiment,
                "news_history": self.news_history,
                "last_update": self.last_update
            }
            self.history_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[SENTIMENT] ⚠️ Error guardando histórico: {e}")


async def main():
    """Test standalone del sentiment manager."""
    manager = SentimentManager()
    
    print("🧠 === SENTIMENT MANAGER TEST ===\n")
    
    # Actualizar sentimiento
    sentiment = await manager.update()
    
    print(f"\n📊 SENTIMIENTO DEL MERCADO:")
    print(f"  Overall: {sentiment['overall'].upper()}")
    print(f"  Score: {sentiment['score']} (-1 a +1)")
    print(f"  Confidence: {sentiment['confidence']}")
    print(f"  Trend: {sentiment['trend']}")
    print(f"  News analyzed: {sentiment['news_count']}")
    
    print(f"\n📰 TOP 3 NOTICIAS MÁS RELEVANTES:")
    for i, news in enumerate(sentiment['recent_events'], 1):
        sent = news['sentiment']
        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(sent['sentiment'], "⚪")
        print(f"\n{i}. {emoji} {news['title']}")
        print(f"   Sentiment: {sent['sentiment']} (score: {sent['score']}, impact: {sent['impact']})")
        print(f"   Source: {news['source']}")


if __name__ == "__main__":
    asyncio.run(main())
