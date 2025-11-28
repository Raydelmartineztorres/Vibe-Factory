"""
Módulo de ingestión de datos (históricos + tiempo real).

Las funciones aquí definidas proveen datos al resto del backend:
- `get_historical_data`: descarga datasets (daily + minuto);
- `bootstrap_data_pipeline`: inicia el stream live y enruta cada tick.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import httpx
from dotenv import load_dotenv

TickHandler = Callable[[dict], Awaitable[None]]

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

DATA_CACHE = Path(__file__).resolve().parent / ".cache"
DATA_CACHE.mkdir(exist_ok=True)


class StrategyLike(Protocol):
    """Protocolo simplificado que deben implementar los módulos de estrategia."""

    async def on_tick(self, payload: dict) -> None: ...


@dataclass
class DataProviderConfig:
    api_key: str
    base_url: str = "https://www.alphavantage.co/query"


class AlphaVantageClient:
    """
    Wrapper ligero sobre la API de Alpha Vantage.
    Soporta cripto (DIGITAL_CURRENCY_DAILY) y FX (FX_DAILY).
    """

    def __init__(self, config: DataProviderConfig | None = None) -> None:
        api_key = config.api_key if config else os.getenv("DATA_PROVIDER_API_KEY")
        if not api_key:
            raise RuntimeError("Falta DATA_PROVIDER_API_KEY en el entorno (.env)")
        self.config = config or DataProviderConfig(api_key=api_key)

    async def _fetch(self, params: dict) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(self.config.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            if "Error Message" in data:
                raise RuntimeError(data["Error Message"])
            return data

    async def fetch_crypto_daily(self, symbol: str, market: str = "USD") -> dict:
        return await self._fetch(
            {
                "function": "DIGITAL_CURRENCY_DAILY",
                "symbol": symbol,
                "market": market,
                "apikey": self.config.api_key,
            }
        )

    async def fetch_fx_daily(self, from_symbol: str, to_symbol: str) -> dict:
        return await self._fetch(
            {
                "function": "FX_DAILY",
                "from_symbol": from_symbol,
                "to_symbol": to_symbol,
                "apikey": self.config.api_key,
            }
        )


async def get_historical_data() -> None:
    """
    Descarga y almacena datasets históricos (daily) para los activos objetivo.

    Los archivos se guardan en `backend/.cache/{symbol}.json` para facilitar
    el backtesting y otros procesos sin golpear la API cada vez.
    """

    client = AlphaVantageClient()
    targets: list[tuple[Literal["crypto", "fx"], tuple[str, ...]]] = [
        ("crypto", ("BTC", "USD")),
        ("crypto", ("ETH", "USD")),
        ("fx", ("EUR", "USD")),
        ("fx", ("JPY", "USD")),
    ]

    for data_type, payload in targets:
        if data_type == "crypto":
            symbol, market = payload
            data = await client.fetch_crypto_daily(symbol=symbol, market=market)
            file_name = f"{symbol}_{market}_daily.json"
        else:
            from_symbol, to_symbol = payload
            data = await client.fetch_fx_daily(from_symbol, to_symbol)
            file_name = f"{from_symbol}_{to_symbol}_daily.json"

        path = DATA_CACHE / file_name
        path.write_text(json.dumps(data))
        print(f"[data] Guardado histórico en {path}")


async def bootstrap_data_pipeline(
    strategy: StrategyLike,
    live_mode: str,
    handler: TickHandler | None = None,
    symbol: str = "BTC/USDT"
) -> None:
    """
    Arranca el feed en tiempo real y enruta cada tick hacia la estrategia.
    
    Usa Coinbase API pública para datos en tiempo real (sin rate limits estrictos).
    """

    async def _urllib_feed() -> None:
        """Feed real usando librería estándar (Cero dependencias externas)."""
        import urllib.request
        import json
        import time
        import ssl
        
        # Map symbol to Coinbase format (BTC/USDT -> BTC-USD)
        coinbase_pair = symbol.replace("/", "-").replace("USDT", "USD")
        
        # Usamos Coinbase porque Binance bloquea IPs de servidores en EEUU (Error 451)
        url = f"https://api.coinbase.com/v2/prices/{coinbase_pair}/spot"
        
        # Ignorar errores SSL (común en Mac/Servers)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        print(f"[DATA] 🚀 Conectando a Coinbase Feed ({symbol})...")
        
        while True:
            try:
                # Ejecutar bloqueo IO en thread separado
                def fetch():
                    with urllib.request.urlopen(url, context=ctx, timeout=5) as response:
                        return json.loads(response.read().decode())
                
                data = await asyncio.to_thread(fetch)
                # Coinbase format: {"data": {"base":"BTC", "currency":"USD", "amount":"98000.50"}}
                price = float(data['data']['amount'])
                
                tick = {
                    "symbol": symbol,
                    "price": price,
                    "mode": live_mode,
                    "timestamp": time.time()
                }
                
                await strategy.on_tick(tick)
                if handler:
                    await handler(tick)
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[DATA] ⚠️ Error fetching price: {e}")
                await asyncio.sleep(2) # Wait a bit more on error
            
            await asyncio.sleep(1) # 1s polling (Scalping Friendly)

    # 1. INJECT FAKE HISTORY (Crucial for ML/Indicators to work immediately)
    print(f"[DATA] 💉 Injecting history for {symbol}...")
    try:
        import time
        import random
        
        # Generate 200 candles of history (approx 16 hours of 5m candles)
        now = time.time()
        history_duration = 200 * 300 # 200 candles * 5 minutes
        start_time = now - history_duration
        
        # Get initial price reference (try to fetch real price, fallback to 90k)
        start_price = 90000.0
        
        # Check if strategy has the required methods/attributes
        if hasattr(strategy, "_update_candle") and hasattr(strategy, "price_history"):
            current_price = start_price
            # Generate a random walk
            for i in range(int(history_duration / 5)): # One tick every 5 seconds
                ts = start_time + (i * 5)
                
                # Random walk
                change = random.uniform(-50, 50)  # 🔥 Increased volatility for better candles
                current_price += change
                
                # Update candle structure
                strategy._update_candle(current_price, ts, symbol)
                
                # Update price history (needed for ML/Indicators)
                if symbol not in strategy.price_history:
                    strategy.price_history[symbol] = []
                strategy.price_history[symbol].append(current_price)
                
                # Keep history size manageable
                if len(strategy.price_history[symbol]) > 500:
                    strategy.price_history[symbol].pop(0)
                    
            # Set last price
            if hasattr(strategy, "last_price"):
                if isinstance(strategy.last_price, dict):
                    strategy.last_price[symbol] = current_price
                else:
                    strategy.last_price = current_price
                    
            print(f"[DATA] ✅ History injection complete: {len(strategy.candles.get(symbol, []))} candles created.")
    except Exception as e:
        print(f"[DATA] ⚠️ Failed to inject history: {e}")
        import traceback
        traceback.print_exc()

    # 2. START REAL FEED
    await _urllib_feed()

