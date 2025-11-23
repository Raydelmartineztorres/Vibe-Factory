"""
Sentiment Analyzer - Analiza el sentimiento de noticias de crypto.

Versión ligera usando keywords (sin necesidad de transformers/torch).
Clasifica noticias como positive, negative, o neutral.
"""

from typing import Dict, List
import re

class SentimentAnalyzer:
    """Analizador de sentimiento basado en keywords."""
    
    # Keywords positivos (bullish)
    POSITIVE_KEYWORDS = {
        # Movimientos de precio positivos
        "surge", "soar", "rally", "climb", "jump", "spike", "rocket", "moon",
        "bull", "bullish", "pump", "breakout", "break", "ath", "all-time high",
        
        # Adopción y crecimiento
        "adoption", "acceptance", "approved", "integrate", "partnership",
        "launch", "announce", "institutional", "mainstream", "breakthrough",
        
        # Sentimiento general positivo
        "optimistic", "confidence", "strong", "growth", "rise", "gain",
        "success", "win", "positive", "upgrade", "improve", "innovation"
    }
    
    # Keywords negativos (bearish)
    NEGATIVE_KEYWORDS = {
        # Movimientos de precio negativos
        "crash", "plunge", "drop", "fall", "tank", "collapse", "dump", "decline",
        "bear", "bearish", "correction", "dip", "down", "lose", "loss",
        
        # Regulación y problemas
        "ban", "banned", "regulation", "lawsuit", "sec", "fine", "fraud",
        "scam", "hack", "hacked", "exploit", "attack", "investigation",
        "fail", "failed", "bankrupt", "shutdown", "suspend",
        
        # Sentimiento general negativo
        "concern", "worried", "fear", "panic", "risk", "danger", "threat",
        "warning", "caution", "negative", "crisis", "volatile", "uncertainty"
    }
    
    # Keywords de alto impacto (multiplican el score)
    HIGH_IMPACT_KEYWORDS = {
        "bitcoin", "btc", "halving", "etf", "institutional", "federal reserve",
        "sec", "regulation", "ban", "adoption", "crash", "ath", "all-time"
    }
    
    def analyze(self, text: str) -> Dict:
        """
        Analiza el sentimiento de un texto.
        
        Args:
            text: Texto a analizar (título + descripción de noticia)
            
        Returns:
            {
                "sentiment": "positive" | "negative" | "neutral",
                "score": float (-1.0 a +1.0),
                "confidence": float (0.0 a 1.0),
                "impact": "high" | "medium" | "low"
            }
        """
        # Normalizar texto
        text_lower = text.lower()
        
        # Contar keywords
        positive_count = self._count_keywords(text_lower, self.POSITIVE_KEYWORDS)
        negative_count = self._count_keywords(text_lower, self.NEGATIVE_KEYWORDS)
        impact_count = self._count_keywords(text_lower, self.HIGH_IMPACT_KEYWORDS)
        
        # Calcular score base
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            # Sin keywords relevantes → neutral
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "impact": "low"
            }
        
        # Score: -1 (muy negativo) a +1 (muy positivo)
        raw_score = (positive_count - negative_count) / total_keywords
        
        # Ajustar score por impacto
        impact_multiplier = 1.0 + (impact_count * 0.2)  # Cada keyword de alto impacto +20%
        adjusted_score = max(-1.0, min(1.0, raw_score * impact_multiplier))
        
        # Determinar sentimiento
        if adjusted_score > 0.2:
            sentiment = "positive"
        elif adjusted_score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Calcular confianza (basado en cantidad de keywords)
        confidence = min(1.0, total_keywords / 5.0)  # Máximo confianza con 5+ keywords
        
        # Determinar impacto
        if impact_count >= 2:
            impact = "high"
        elif impact_count == 1:
            impact = "medium"
        else:
            impact = "low"
        
        return {
            "sentiment": sentiment,
            "score": round(adjusted_score, 2),
            "confidence": round(confidence, 2),
            "impact": impact,
            "_debug": {
                "positive_keywords": positive_count,
                "negative_keywords": negative_count,
                "impact_keywords": impact_count
            }
        }
    
    def analyze_news_item(self, news: Dict) -> Dict:
        """
        Analiza el sentimiento de un item de noticia completo.
        
        Args:
            news: Dict con campos 'title' y 'description'
            
        Returns:
            News dict con campo 'sentiment' añadido
        """
        # Combinar título y descripción (título pesa más)
        title = news.get("title", "")
        description = news.get("description", "")
        combined_text = f"{title} {title} {description}"  # Título aparece 2 veces (mayor peso)
        
        # Analizar sentimiento
        sentiment_result = self.analyze(combined_text)
        
        # Añadir al news dict
        news["sentiment"] = sentiment_result
        
        return news
    
    def _count_keywords(self, text: str, keywords: set) -> int:
        """Cuenta cuántas keywords aparecen en el texto."""
        count = 0
        for keyword in keywords:
            # Usar regex para match de palabra completa
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = re.findall(pattern, text)
            count += len(matches)
        return count
    
    def get_sentiment_emoji(self, sentiment: str) -> str:
        """Retorna emoji según sentimiento."""
        emoji_map = {
            "positive": "🟢",
            "negative": "🔴",
            "neutral": "🟡"
        }
        return emoji_map.get(sentiment, "⚪")


# Test standalone
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Test cases
    test_texts = [
        "Bitcoin surges past $90k as institutional adoption grows",
        "Crypto market crashes amid SEC regulation fears",
        "BTC price stable around $87k, traders remain cautious",
        "Major exchange announces Bitcoin ETF approval - bullish news!",
        "Hackers exploit DeFi protocol, millions lost in attack"
    ]
    
    print("=== SENTIMENT ANALYZER TEST ===\n")
    for text in test_texts:
        result = analyzer.analyze(text)
        emoji = analyzer.get_sentiment_emoji(result["sentiment"])
        
        print(f"{emoji} {result['sentiment'].upper()} (score: {result['score']}, confidence: {result['confidence']})")
        print(f"   Text: {text}")
        print(f"   Impact: {result['impact']}")
        print()
