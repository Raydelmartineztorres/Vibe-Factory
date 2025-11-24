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

    Parameters
    ----------
    strategy:
        Instancia con método `on_tick`.
    live_mode:
        "demo" (paper) o "real".
    handler:
        Callback opcional para observar cada tick (logging/metrics).
    symbol:
        Símbolo a trackear (ej: "BTC/USDT").
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
        
        print(f"[DATA] Conectando a Coinbase (Cloud Friendly)...")
        
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
                print(f"[DATA] Error fetching price: {e}")
            
            await asyncio.sleep(1)

    # Iniciar el feed correspondiente
    if live_mode == "real" or live_mode == "demo":
        await _urllib_feed()
    else:
        await _fake_feed()

    async def _real_feed() -> None:
        """Polling real a Alpha Vantage (respetando rate limits)."""
        client = AlphaVantageClient()
        # Alpha Vantage Free Tier: 5 calls/minute.
        # Polling cada 60s para estar seguros.
        
        symbol = "BTC"
        market = "USD"
        
        while True:
            try:
                # Usamos la API de Daily como proxy de "live" por ahora, 
                # ya que la API intraday puede ser más restrictiva o de pago.
                # Para simular "live", tomamos el último cierre.
                data = await client.fetch_crypto_daily(symbol, market)
                
                # Extraer el último dato
                ts = data.get("Time Series (Digital Currency Daily)", {})
                if not ts:
                    print("[data] No data received")
                    await asyncio.sleep(60)
                    continue
                    
                last_date = sorted(ts.keys())[-1]
                last_candle = ts[last_date]
                # Determine the correct close price key (Alpha Vantage may use 4a or 4b)
                close_key = next((k for k in last_candle.keys() if "close" in k.lower()), None)
                if close_key is None:
                    raise KeyError("Close price key not found in Alpha Vantage response")
                current_price = float(last_candle[close_key])
                
                tick = {
                    "symbol": f"{symbol}/{market}",
                    "price": current_price,
                    "mode": live_mode,
                    "timestamp": last_date
                }
                
                await strategy.on_tick(tick)
                if handler:
                    await handler(tick)
                    
                print(f"[data] Tick real: {tick['symbol']} @ {tick['price']}")
                
            except Exception as e:
                print(f"[data] Error fetching real data: {e}")
            
            await asyncio.sleep(70) # 70s para evitar rate limit (5 req/min)

    if live_mode == "real":
        print("[data] Iniciando modo REAL con Alpha Vantage...")
        await _real_feed()
    else:
        print("[data] Iniciando modo DEMO (simulado)...")
        await _fake_feed()

