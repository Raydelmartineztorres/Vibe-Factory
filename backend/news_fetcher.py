"""
News Fetcher - Descarga noticias de criptomonedas de múltiples fuentes.

Características:
- Múltiples fuentes de noticias gratuitas
- Cache para evitar requests duplicados
- Filtrado por relevancia (BTC/crypto)
- Actualización cada 15 minutos
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import json
import httpx

class NewsFetcher:
    """Fetcher de noticias de crypto desde APIs gratuitas."""
    
    def __init__(self):
        self.cache_file = Path(__file__).parent / ".cache" / "crypto_news.json"
        self.cache_ttl = 900  # 15 minutos en segundos
        self.last_fetch = 0
        self.cached_news = []
        
    async def fetch_latest_news(self, limit: int = 10) -> List[Dict]:
        """
        Descarga las últimas noticias de crypto.
        
        Args:
            limit: Número máximo de noticias a retornar
            
        Returns:
            Lista de noticias con formato estandarizado
        """
        # Verificar cache
        if self._is_cache_valid():
            print("[NEWS] 📰 Usando noticias en cache")
            return self.cached_news[:limit]
        
        print("[NEWS] 🌐 Descargando noticias frescas...")
        
        # Intentar múltiples fuentes
        news = []
        
        # Fuente 1: CryptoCompare (gratuita, no requiere API key)
        try:
            crypto_compare_news = await self._fetch_from_cryptocompare(limit)
            news.extend(crypto_compare_news)
        except Exception as e:
            print(f"[NEWS] ⚠️ CryptoCompare falló: {e}")
        
        # Fuente 2: CoinGecko (gratuita, no requiere API key)
        try:
            coingecko_news = await self._fetch_from_coingecko(limit)
            news.extend(coingecko_news)
        except Exception as e:
            print(f"[NEWS] ⚠️ CoinGecko falló: {e}")
        
        # Deduplicar por título
        unique_news = self._deduplicate_news(news)
        
        # Ordenar por fecha (más recientes primero)
        unique_news.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        
        # Guardar en cache
        self.cached_news = unique_news[:limit]
        self._save_cache()
        
        print(f"[NEWS] ✅ Descargadas {len(self.cached_news)} noticias únicas")
        return self.cached_news
    
    async def _fetch_from_cryptocompare(self, limit: int) -> List[Dict]:
        """Descarga noticias desde CryptoCompare."""
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            news_items = data.get("Data", [])[:limit]
            
            # Convertir a formato estándar
            standardized = []
            for item in news_items:
                standardized.append({
                    "id": item.get("id"),
                    "title": item.get("title", ""),
                    "description": item.get("body", "")[:200],  # Primeros 200 chars
                    "source": item.get("source_info", {}).get("name", "CryptoCompare"),
                    "published_at": datetime.fromtimestamp(
                        item.get("published_on", 0), 
                        tz=timezone.utc
                    ).isoformat(),
                    "url": item.get("url", ""),
                    "image_url": item.get("imageurl", ""),
                    "categories": item.get("categories", ""),
                    "sentiment": None  # Se calculará después
                })
            
            return standardized
    
    async def _fetch_from_coingecko(self, limit: int) -> List[Dict]:
        """Descarga noticias desde CoinGecko Status Updates."""
        # CoinGecko tiene un feed de status updates que funciona sin API key
        url = "https://api.coingecko.com/api/v3/status_updates"
        params = {
            "category": "general",
            "per_page": limit
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            updates = data.get("status_updates", [])
            
            # Convertir a formato estándar
            standardized = []
            for item in updates:
                standardized.append({
                    "id": f"cg_{item.get('user', '')}_{item.get('created_at', '')}",
                    "title": item.get("user_title", "Crypto Update"),
                    "description": item.get("description", "")[:200],
                    "source": "CoinGecko",
                    "published_at": item.get("created_at", ""),
                    "url": "https://www.coingecko.com",
                    "image_url": "",
                    "categories": item.get("category", ""),
                    "sentiment": None
                })
            
            return standardized
    
    def _deduplicate_news(self, news: List[Dict]) -> List[Dict]:
        """Elimina noticias duplicadas basado en similitud de títulos."""
        seen_titles = set()
        unique = []
        
        for item in news:
            # Normalizar título (lowercase, sin espacios extra)
            title = item.get("title", "").lower().strip()
            
            # Crear un hash simple del título (primeras 50 chars)
            title_hash = title[:50]
            
            if title_hash and title_hash not in seen_titles:
                seen_titles.add(title_hash)
                unique.append(item)
        
        return unique
    
    def _is_cache_valid(self) -> bool:
        """Verifica si el cache es válido (no expiró)."""
        if not self.cached_news:
            return False
        
        time_since_fetch = time.time() - self.last_fetch
        return time_since_fetch < self.cache_ttl
    
    def _save_cache(self):
        """Guarda noticias en cache local."""
        try:
            self.cache_file.parent.mkdir(exist_ok=True)
            cache_data = {
                "news": self.cached_news,
                "fetched_at": time.time()
            }
            self.cache_file.write_text(json.dumps(cache_data, indent=2))
            self.last_fetch = time.time()
        except Exception as e:
            print(f"[NEWS] ⚠️ Error guardando cache: {e}")


async def main():
    """Test standalone del news fetcher."""
    fetcher = NewsFetcher()
    news = await fetcher.fetch_latest_news(limit=5)
    
    print(f"\n📰 === ÚLTIMAS {len(news)} NOTICIAS DE CRYPTO ===\n")
    for i, item in enumerate(news, 1):
        print(f"{i}. {item['title']}")
        print(f"   Source: {item['source']} | {item['published_at']}")
        print(f"   {item['description'][:100]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
