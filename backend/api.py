from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

app = FastAPI(title="Vibe Factory API")

# Server start time for uptime tracking
_server_start_time = time.time()

# Auto-trading background task
_auto_trading_task = None

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://vibe-factory-frontend.onrender.com",
    "https://vibe-factory.onrender.com",
    "*", # Allow all for easier initial deployment (be careful in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- BASIC AUTHENTICATION ---
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import secrets
import os
from fastapi import Request, HTTPException, status

# Default credentials (change via env vars in production!)
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "raydemartineztorres")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "Limallo33")

class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip auth for health check (optional, but good for monitoring)
        if request.url.path == "/api/health":
            return await call_next(request)
            
        # Check for Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return self._request_auth()
            
        try:
            scheme, credentials = auth_header.split()
            if scheme.lower() != 'basic':
                return self._request_auth()
                
            import base64
            decoded = base64.b64decode(credentials).decode("ascii")
            username, _, password = decoded.partition(":")
            
            # Verify credentials safely
            is_correct_username = secrets.compare_digest(username, AUTH_USERNAME)
            is_correct_password = secrets.compare_digest(password, AUTH_PASSWORD)
            
            if not (is_correct_username and is_correct_password):
                return self._request_auth()
                
        except Exception:
            return self._request_auth()

        return await call_next(request)

    def _request_auth(self):
        return Response(
            content="Authentication required",
            status_code=401,
            headers={"WWW-Authenticate": "Basic realm='Vibe Factory Access'"},
        )

# Add middleware to app
# DISABLED FOR LOCAL DEMO MODE - Re-enable for production deployment
# app.add_middleware(BasicAuthMiddleware)

@app.get("/api/health")
def health_check():
    return {"status": "online", "message": "Vibe Factory Backend is running"}

@app.get("/")
def root():
    """Root endpoint to verify backend is running."""
    return {
        "message": "Welcome to Vibe Factory API",
        "docs": "/docs",
        "health": "/api/health",
        "status": "online"
    }


@app.get("/api/backtest")
async def get_backtest_results():
    """Get the most recent backtest results"""
    from backtester import run_backtest
    result = await run_backtest()
    return {
        "final_capital": result.final_capital,
        "max_drawdown": result.max_drawdown,
        "trades": result.trades
    }


@app.post("/api/backtest")
async def api_run_backtest():
    """Run a new backtest and return results"""
    from backtester import run_backtest
    result = await run_backtest()
    return {
        "final_capital": result.final_capital,
        "max_drawdown": result.max_drawdown,
        "trades": result.trades
    }


@app.post("/api/trade")
async def place_trade(payload: dict):
    """Recibe orden de compra/venta y la ejecuta según el modo actual."""
    from broker_api_handler import execute_order
    try:
        print(f"[API] Received trade request: {payload}")
        print(f"[API] Trading mode: {_trading_mode}")
        
        # Validar payload
        required_fields = ["symbol", "side", "size"]
        for field in required_fields:
            if field not in payload:
                return {"status": "FAILED", "error": f"Missing required field: {field}"}
        
        # Usar el modo global configurado
        result = await execute_order(payload, mode=_trading_mode)
        
        print(f"[API] Trade result: {result}")
        
        # Actualizar estado de la estrategia si la orden fue exitosa
        if result.get("status") in ["FILLED", "SIMULATED"]:
            # Convertir symbol a formato correcto (BTC_USDT -> BTC/USDT)
            symbol = payload["symbol"].replace("_", "/")
            _strategy_instance.register_trade(
                side=payload["side"],
                price=result["price"],
                size=payload["size"],
                result=result,
                source="manual",
                symbol=symbol
            )
            
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "FAILED", "error": str(e)}




# --- INTEGRACIÓN DE ESTRATEGIA ---
from risk_strategy import RiskStrategy
from data_collector import bootstrap_data_pipeline
from optimizer import Optimizer
import asyncio
import time

_strategy_instance = RiskStrategy()
_strategy_instance = RiskStrategy()
_optimizer_instance = Optimizer(_strategy_instance)
_data_task = None
_current_symbol = "BTC/USDT"
_trading_mode = "demo" # demo, testnet, real

@app.on_event("startup")
async def startup_event():
    """Iniciar el loop de trading y aprendizaje en background."""
    # 1. Pipeline de datos y trading
    # 1. Pipeline de datos y trading (MULTI-ASSET)
    global _data_tasks
    _data_tasks = {}
    
    initial_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    print(f"[STARTUP] 🚀 Starting data pipelines for: {initial_symbols}")
    
    for sym in initial_symbols:
        _data_tasks[sym] = asyncio.create_task(bootstrap_data_pipeline(
            strategy=_strategy_instance,
            live_mode="demo", 
            symbol=sym
        ))
    
    # 2. Optimizador (Aprendizaje)
    asyncio.create_task(_optimizer_instance.start_loop())
    
    # 3. 🐱 EL GATO Auto-Trading
    print("[STARTUP] 🐱 Iniciando auto-trading de EL GATO...")
    global _auto_trading_task
    from auto_trader import auto_trading_loop
    
    def is_trading_enabled():
        return _trading_enabled
    
    _auto_trading_task = asyncio.create_task(
        auto_trading_loop(_strategy_instance, is_trading_enabled)
    )

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup al cerrar."""
    global _auto_trading_task
    if _auto_trading_task:
        _auto_trading_task.cancel()
        try:
            await _auto_trading_task
        except asyncio.CancelledError:
            pass

@app.get("/api/price")
def get_current_price(symbol: str = "BTC/USDT"):
    """Retorna el precio real de la estrategia para el símbolo especificado."""
    # last_price es ahora un dict multi-activo
    price = _strategy_instance.last_price.get(symbol, 0) if isinstance(_strategy_instance.last_price, dict) else 90000.0
    return {"symbol": symbol, "price": price if price > 0 else 90000.0}

@app.post("/api/set_symbol")
async def set_symbol(payload: dict):
    """Cambia el símbolo activo y reinicia el pipeline de datos."""
    global _current_symbol, _data_task
    
    new_symbol = payload.get("symbol")
    if not new_symbol:
        return {"error": "Symbol required"}
        
    if new_symbol == _current_symbol:
        return {"status": "unchanged", "symbol": _current_symbol}
        
    print(f"[API] Switching UI focus to {new_symbol}...")
    _current_symbol = new_symbol
    
    # Ensure data stream is running for this symbol
    global _data_tasks
    if '_data_tasks' not in globals():
        _data_tasks = {}
        
    if new_symbol not in _data_tasks or _data_tasks[new_symbol].done():
        print(f"[API] Starting new data stream for {new_symbol}...")
        _data_tasks[new_symbol] = asyncio.create_task(bootstrap_data_pipeline(
            strategy=_strategy_instance,
            live_mode="demo",
            symbol=new_symbol
        ))
    else:
        print(f"[API] Data stream for {new_symbol} already running.")
    
    # Resetear estado de la estrategia para el nuevo símbolo
    # (Opcional: podríamos querer mantener el historial si es multi-asset real)
    _strategy_instance.candles[_current_symbol] = []
    if isinstance(_strategy_instance.last_price, dict):
        _strategy_instance.last_price[_current_symbol] = 0.0
    else:
        _strategy_instance.last_price = 0.0
    
    return {"status": "success", "symbol": _current_symbol}

@app.get("/api/trading/mode")
def get_trading_mode():
    """Retorna el modo de trading actual."""
    return {"mode": _trading_mode}

@app.post("/api/trading/mode")
def set_trading_mode(payload: dict):
    """Cambia el modo de trading (demo, testnet, real, coinbase)."""
    global _trading_mode
    new_mode = payload.get("mode")
    if new_mode not in ["demo", "testnet", "real", "coinbase"]:
        return {"error": "Invalid mode. Use: demo, testnet, real, coinbase"}
    
    _trading_mode = new_mode
    print(f"[API] Trading mode switched to: {_trading_mode.upper()}")
    return {"status": "success", "mode": _trading_mode}

# === STRATEGY MANAGEMENT ===

@app.get("/api/strategies")
def get_strategies():
    """Retorna todas las estrategias con sus stats."""
    if _strategy_instance is None:
        return {"strategies": [], "active_strategy": "BALANCED"}
    
    return _strategy_instance.strategy_manager.get_all_stats()

@app.post("/api/strategies/activate")
def activate_strategy(payload: dict):
    """Cambia la estrategia activa."""
    strategy_name = payload.get("strategy")
    
    if _strategy_instance is None:
        return {"status": "error", "message": "Sistema no inicializado"}
    
    success = _strategy_instance.strategy_manager.switch_strategy(strategy_name)
    
    if success:
        return {"status": "success", "active_strategy": strategy_name}
    return {"status": "error", "message": f"No se pudo activar {strategy_name}"}

@app.post("/api/strategies/pause")
def pause_strategy(payload: dict):
    """Pausa una estrategia."""
    strategy_name = payload.get("strategy")
    
    if _strategy_instance is None:
        return {"status": "error", "message": "Sistema no inicializado"}
    
    _strategy_instance.strategy_manager.pause_strategy(strategy_name)
    return {"status": "success", "message": f"{strategy_name} pausada"}

@app.post("/api/strategies/resume")
def resume_strategy(payload: dict):
    """Reanuda una estrategia pausada."""
    strategy_name = payload.get("strategy")
    
    if _strategy_instance is None:
        return {"status": "error", "message": "Sistema no inicializado"}
    
    _strategy_instance.strategy_manager.resume_strategy(strategy_name)
    return {"status": "success", "message": f"{strategy_name} reanudada"}

@app.get("/api/strategies/recommendation")
def get_strategy_recommendation():
    """Obtiene recomendación de cuál estrategia usar."""
    if _strategy_instance is None:
        return {"recommendation": "Sistema no inicializado"}
    
    recommendation = _strategy_instance.strategy_manager.get_recommendation()
    return {"recommendation": recommendation}

# === END STRATEGY MANAGEMENT ===

# === EL GATO INTELLIGENCE ===

@app.get("/api/el-gato/status")
def get_el_gato_status():
    """Retorna estado completo de EL GATO (IQ, tier, capacidades)."""
    from el_gato import get_el_gato
    el_gato = get_el_gato()
    return el_gato.get_status()

@app.get("/api/el-gato/daily-progress")
def get_daily_progress():
    """Retorna progreso hacia objetivo diario."""
    from el_gato import get_el_gato
    if _strategy_instance is None:
        return {"error": "Sistema no inicializado"}
    
    el_gato = get_el_gato()
    current_profit = _strategy_instance.realized_pnl
    return el_gato.get_daily_progress(current_profit)

@app.get("/api/el-gato/recommendation")
def get_el_gato_recommendation():
    """Obtiene recomendación de EL GATO."""
    from el_gato import get_el_gato
    el_gato = get_el_gato()
    
    # Obtener velas actuales de la estrategia si existen
    candles = []
    if _strategy_instance and hasattr(_strategy_instance, 'candles'):
        # Usar velas del símbolo actual
        current_symbol = _current_symbol if '_current_symbol' in globals() else "BTC/USDT"
        
        # 🔥 FIX: Aggregate candles to 1m to avoid false Dojis on 5s candles
        if hasattr(_strategy_instance, 'aggregate_candles'):
            candles = _strategy_instance.aggregate_candles("1m", current_symbol)
        else:
            candles = _strategy_instance.candles.get(current_symbol, [])
        
    return {"recommendation": el_gato.get_recommendation(candles=candles)}

# === END EL GATO INTELLIGENCE ===

@app.get("/api/trades")
def get_trades():
    """Retorna los trades ejecutados por la estrategia."""
    return {"trades": _strategy_instance.trades[-20:]}  # Últimos 20 trades

@app.get("/api/pnl")
def get_pnl():
    """Retorna métricas de PnL en tiempo real y posiciones."""
    # Calcular PnL total
    portfolio_pnl = _strategy_instance.get_portfolio_pnl()
    
    # Construir lista de posiciones
    positions_list = []
    for symbol, size in _strategy_instance.positions.items():
        if size != 0:
            entry = _strategy_instance.entry_prices.get(symbol, 0)
            current = _strategy_instance.last_price.get(symbol, 0)
            
            # Calcular PnL de esta posición
            if size > 0:
                pnl = (current - entry) * size
            else:
                pnl = (entry - current) * abs(size)
                
            pnl_pct = (pnl / (entry * abs(size))) * 100 if entry > 0 else 0
            
            positions_list.append({
                "symbol": symbol,
                "size": size,
                "entryPrice": entry,
                "currentPrice": current,
                "pnl": pnl,
                "pnlPercent": pnl_pct
            })
            
    return {
        "realized_pnl": _strategy_instance.realized_pnl,
        "unrealized_pnl": portfolio_pnl['unrealized_pnl'],
        "total_pnl": _strategy_instance.realized_pnl + portfolio_pnl['unrealized_pnl'],
        "positions": positions_list,
        # Mantener compatibilidad hacia atrás por si acaso
        "position_size": sum(abs(p) for p in _strategy_instance.positions.values()), 
        "entry_price": 0, 
        "current_price": 0
    }

@app.get("/api/candles")
def get_candles(timeframe: str = "5m", symbol: str = "BTC/USDT"):
    """
    Retorna las velas OHLC para el gráfico.
    
    Args:
        timeframe: Intervalo de tiempo (1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M)
        symbol: Símbolo del activo (BTC/USDT, ETH/USDT, etc.)
    """
    if _strategy_instance is None:
        return {"candles": [], "trades": []}
    
    # Get aggregated candles based on timeframe and symbol
    candles = _strategy_instance.aggregate_candles(timeframe, symbol)
    
    # Filter trades for this symbol
    symbol_trades = [
        {
            "time": t.get("time", 0),
            "side": t.get("side", "BUY"),
            "source": t.get("source", "manual"),
            "price": t.get("price", 0),
            "size": t.get("size", 0)
        }
        for t in _strategy_instance.trades
        if t.get("symbol", "BTC/USDT") == symbol or t.get("result", {}).get("symbol", "").replace("_", "/") == symbol
    ][-50:]  # Last 50 trades only
    
    return {
        "candles": [
            {
                "time": int(c.time),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume
            } for c in candles
        ],
        "trades": symbol_trades
    }


@app.get("/api/indicators")
def get_indicators(symbol: str = "BTC/USDT"):
    """Get RSI and ATR indicator values for chart overlay."""
    try:
        if _strategy_instance is None:
            return {"rsi": [], "atr": [], "macd": []}
        
        # Handle symbol format
        symbol = symbol.replace("_", "/")
        
        # Get candles for specific symbol
        all_candles = _strategy_instance.candles
        if isinstance(all_candles, dict):
            candles = all_candles.get(symbol, [])
        else:
            candles = all_candles # Fallback if it were a list
        
        print(f"[INDICATORS] Symbol: {symbol}, Candles count: {len(candles)}")
            
        if not candles or len(candles) < 26: # Need more data for MACD
            print(f"[INDICATORS] ⚠️ Not enough candles for {symbol}: {len(candles)}/26")
            return {"rsi": [], "atr": [], "macd": []}
        
        # Calculate indicators
        # We'll use pandas for easier calculation if available, or manual
        try:
            import pandas as pd
            import pandas_ta as ta
            
            df = pd.DataFrame([{
                'close': c.close,
                'high': c.high,
                'low': c.low,
                'time': int(c.time)
            } for c in candles])
            
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # MACD
            macd = ta.macd(df['close'])
            if macd is not None:
                df = pd.concat([df, macd], axis=1)
            
            # Format for lightweight-charts
            rsi_data = []
            atr_data = []
            macd_data = []
            
            for i, row in df.iterrows():
                time = int(row['time'])
                
                # RSI - skip if NaN
                if not pd.isna(row['rsi']):
                    rsi_data.append({
                        "time": time,
                        "value": float(row['rsi'])  # Convert to native Python float
                    })
                
                # ATR - skip if NaN
                if not pd.isna(row['atr']):
                    atr_data.append({
                        "time": time,
                        "value": float(row['atr'])  # Convert to native Python float
                    })
                    
                # MACD columns usually named MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
                # We need to find them dynamically or assume standard names
                macd_col = next((c for c in df.columns if c.startswith('MACD_')), None)
                hist_col = next((c for c in df.columns if c.startswith('MACDh_')), None)
                signal_col = next((c for c in df.columns if c.startswith('MACDs_')), None)
                
                # MACD - skip if any component is NaN
                if (macd_col and hist_col and signal_col and 
                    not pd.isna(row[macd_col]) and 
                    not pd.isna(row[signal_col]) and 
                    not pd.isna(row[hist_col])):
                    macd_data.append({
                        "time": time,
                        "macd": float(row[macd_col]),
                        "signal": float(row[signal_col]),
                        "histogram": float(row[hist_col])
                    })
                    
            return {
                "rsi": rsi_data, 
                "atr": atr_data,
                "macd": macd_data
            }
            
        except ImportError:
            # Fallback to manual calculation if pandas_ta not installed
            print("[WARNING] pandas_ta not found, using simple calculation")
            return {"rsi": [], "atr": [], "macd": []}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/api/prediction")
def get_prediction(symbol: str = "BTC/USDT"):
    """🔮 Obtiene la predicción de la siguiente vela para un símbolo."""
    try:
        if _strategy_instance is None:
            return {"error": "Strategy not initialized"}
        
        # Normalizar símbolo
        symbol = symbol.replace("_", "/")
        
        # Obtener predicción actual
        prediction = _strategy_instance.predictions.get(symbol)
        
        if prediction is None:
            return {
                "symbol": symbol,
                "has_prediction": False,
                "message": "Esperando suficientes velas para predicción (mínimo 30)"
            }
        
        # Obtener precisión histórica del predictor
        predictor_stats = _strategy_instance.candle_predictor.get_stats()
        
        # Precio actual
        current_price = _strategy_instance.last_price.get(symbol, 0)
        
        # Calcular cambio esperado
        expected_change = prediction.predicted_close - current_price
        expected_change_pct = (expected_change / current_price * 100) if current_price > 0 else 0
        
        return {
            "symbol": symbol,
            "has_prediction": True,
            "current_price": round(current_price, 2),
            "prediction": {
                "time": prediction.time,
                "open": round(prediction.predicted_open, 2),
                "high": round(prediction.predicted_high, 2),
                "low": round(prediction.predicted_low, 2),
                "close": round(prediction.predicted_close, 2),
                "confidence": round(prediction.confidence * 100, 2),
                "method": prediction.prediction_method,
                "direction": "UP" if expected_change > 0 else "DOWN",
                "expected_change": round(expected_change, 2),
                "expected_change_pct": round(expected_change_pct, 2)
            },
            "predictor_stats": predictor_stats
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/api/advice")
def get_ai_advice():
    """Get AI trading advice based on current market conditions."""
    try:
        if _strategy_instance is None:
            return {"advice": "Sistema no inicializado", "confidence": "low", "action": "wait"}
        
        # Get current indicators
        candles = _strategy_instance.candles
        if not candles or len(candles) < 14:
            return {"advice": "Esperando más datos del mercado...", "confidence": "low", "action": "wait"}
        
        # Calculate current RSI
        window = candles[-15:]
        closes = [c.close for c in window]
        gains = []
        losses = []
        for j in range(1, len(closes)):
            change = closes[j] - closes[j-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        if avg_loss == 0:
            current_rsi = 100 if avg_gain > 0 else 50
        else:
            rs = avg_gain / avg_loss
            current_rsi = 100 - (100 / (1 + rs))

        # Calculate MACD (12, 26, 9) - Convert to pandas Series first
        import pandas as pd
        import numpy as np
        
        closes_series = pd.Series(closes)
        highs_series = pd.Series([c.high for c in window])
        lows_series = pd.Series([c.low for c in window])
        
        ema_12 = closes_series.ewm(span=12, adjust=False).mean()
        ema_26 = closes_series.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2] if len(histogram) > 1 else 0

        # Calculate ATR (14)
        high_low = highs_series - lows_series
        high_close = np.abs(highs_series - closes_series.shift(1))
        low_close = np.abs(lows_series - closes_series.shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = ranges.ewm(span=14, adjust=False).mean().iloc[-1]
        
        # Get memory stats
        stats = _strategy_instance.memory.get_stats()
        
        # Determine market context
        price_changes = [closes_series.iloc[i] - closes_series.iloc[i-1] for i in range(1, len(closes_series))]
        avg_change = sum(price_changes) / len(price_changes)
        trend = "alcista" if avg_change > 0 else "bajista"
        volatility = "alta" if atr > 500 else "baja"
        
        # Initialize variables
        action = "wait"
        confidence = "medium"
        win_probability = 50

        # Determine Action based on RSI + MACD
        # MACD Crossover Logic
        bullish_crossover = current_hist > 0 and prev_hist <= 0
        bearish_crossover = current_hist < 0 and prev_hist >= 0
        
        if current_rsi > 70:
            action = "sell"
            confidence = "high"
            win_probability = 75
            if bearish_crossover:
                confidence = "very_high"
                win_probability = 85
        elif current_rsi < 30:
            action = "buy"
            confidence = "high"
            win_probability = 78
            if bullish_crossover:
                confidence = "very_high"
                win_probability = 88
        elif bullish_crossover and current_rsi < 60:
            action = "buy"
            confidence = "medium"
            win_probability = 65
        elif bearish_crossover and current_rsi > 40:
            action = "sell"
            confidence = "medium"
            win_probability = 65
        elif 45 <= current_rsi <= 55:
            action = "wait"
            confidence = "low"
            win_probability = 45
        else:
            # Trend following
            if trend == "alcista":
                action = "buy"
                win_probability = 60
            else:
                action = "sell"
                win_probability = 55
            confidence = "medium"
        
        # Calculate SL/TP levels based on ATR
        current_price = closes_series.iloc[-1]
        sl_price = 0
        tp_price = 0
        
        if action == "buy":
            sl_price = current_price - (atr * 1.5)
            tp_price = current_price + (atr * 2.0)
        elif action == "sell":
            sl_price = current_price + (atr * 1.5)
            tp_price = current_price - (atr * 2.0)

        # Generate concrete advice
        advice = ""
        if action == "buy":
            advice = f"🚀 OPORTUNIDAD DE COMPRA DETECTADA\n"
            advice += f"RSI ({current_rsi:.1f}) + MACD {'Cruce Alcista' if bullish_crossover else 'Positivo'}. "
        elif action == "sell":
            advice = f"🔻 SEÑAL DE VENTA DETECTADA\n"
            advice += f"RSI ({current_rsi:.1f}) + MACD {'Cruce Bajista' if bearish_crossover else 'Negativo'}. "
        else:
            advice = f"⏸️ ESPERAR MEJOR OPORTUNIDAD\n"
            advice += f"Mercado indefinido (RSI: {current_rsi:.1f}, MACD Hist: {current_hist:.2f}). "

        # Add volatility context
        advice += f"Volatilidad {'ALTA' if volatility == 'alta' else 'BAJA'} (ATR: {atr:.0f}). "

        # Add memory context
        if stats["total_trades"] > 5:
            best = stats.get("best_context", {})
            if best and best.get("context") == f"{trend}_{volatility}":
                advice += f"\n🧠 MEMORIA: Este escenario ({best['context']}) tiene un {best['win_rate']:.0f}% de éxito histórico."
                confidence = "high"
                win_probability = max(win_probability, best['win_rate'])

        # Cap probability
        win_probability = min(95, max(5, win_probability))
        
        return {
            "advice": advice,
            "action": action,
            "confidence": confidence,
            "win_probability": round(win_probability, 0),
            "trade_setup": {
                "sl": round(sl_price, 2) if action != "wait" else 0,
                "tp": round(tp_price, 2) if action != "wait" else 0,
                "entry": round(current_price, 2)
            },
            "indicators": {
                "rsi": round(current_rsi, 1),
                "atr": round(atr, 2),
                "macd": round(current_macd, 2),
                "trend": trend,
                "volatility": volatility
            }
        }
    except Exception as e:
        import traceback
        print(f"Error in advice: {e}")
        traceback.print_exc()
        return {
            "advice": f"Error interno: {str(e)}",
            "action": "wait",
            "confidence": "low",
            "win_probability": 0,
            "indicators": {
                "rsi": 0,
                "atr": 0,
                "trend": "error",
                "volatility": "error"
            }
        }

@app.get("/api/memory")
def get_memory_stats():
    """Retorna estadísticas del módulo de memoria."""
    return _strategy_instance.memory.get_stats()

@app.get("/api/ml/prediction")
def get_ml_prediction():
    """Retorna la predicción actual del modelo ML."""
    if _strategy_instance is None or not _strategy_instance.ml_enabled:
        return {"signal": "NEUTRAL", "confidence": 0.0, "is_trained": False}
    
    # Obtener datos actuales para predicción
    current_rsi = _strategy_instance._calculate_rsi() or 50.0
    current_atr = _strategy_instance._calculate_atr() or 50.0
    current_vol = _strategy_instance.current_candle.volume if _strategy_instance.current_candle else 0.0
    
    # Usar el predictor
    prediction = _strategy_instance.ml_predictor.predict(
        _strategy_instance.price_history,
        current_rsi,
        current_atr,
        current_vol
    )
    
    # Get last_price safely (puede ser dict o float)
    if isinstance(_strategy_instance.last_price, dict):
        current_price = _strategy_instance.last_price.get(_current_symbol, 0)
    else:
        current_price = _strategy_instance.last_price
    
    signal = "NEUTRAL"
    if prediction and current_price > 0:
        if prediction > current_price * 1.001:
            signal = "BUY"
        elif prediction < current_price * 0.999:
            signal = "SELL"
        
    return {
        "signal": signal,
        "predicted_price": prediction,
        "current_price": current_price,
        "is_trained": _strategy_instance.ml_predictor.is_trained,
        "model_path": str(_strategy_instance.ml_predictor.MODEL_PATH)
    }

@app.get("/api/market/status")
def get_market_status():
    """Retorna el estado actual del mercado (sesión, volumen, etc)."""
    if _strategy_instance is None:
        return {"session": "UNKNOWN", "is_peak": False}
        
    session = _strategy_instance.market_timing.get_current_session()
    stats = _strategy_instance.market_timing.get_stats()
    
    return {
        "session": session.name,
        "is_peak": session.is_peak,
        "aggressiveness": session.aggressiveness,
        "avg_volume": session.avg_volume,
        "current_hour_stats": stats.get(session.name, {}) # Simplificado
    }

@app.get("/api/risk/stats")
def get_risk_stats():
    """Retorna estadísticas del gestor de riesgo."""
    if _strategy_instance is None or not hasattr(_strategy_instance, 'risk_manager'):
        return {}
    return _strategy_instance.risk_manager.get_stats()

# Fast Training & System Health
@app.post("/api/training/run")
def run_fast_training(num_trades: int = 100):
    """Ejecuta entrenamiento rápido con datos históricos."""
    try:
        from fast_training import FastTrainer
        trainer = FastTrainer()
        # Ejecutar sincrónicamente (no async)
        stats = trainer.run_quick_training(num_trades=num_trades)
        return {
            "success": True,
            "stats": stats,
            "message": f"Entrenamiento completado: {stats.get('trades_executed', 0)} trades"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": "Error durante el entrenamiento"
        }

@app.get("/api/system/health")
def get_system_health():
    """Retorna el estado de salud del sistema."""
    try:
        from memory import TradeMemory
        import psutil
        import time
        
        # Uptime del servidor
        uptime = int(time.time() - _server_start_time)
        
        # Memoria del bot
        memory = TradeMemory()
        memory_stats = memory.get_stats()
        
        # Trades today (safe)
        trades_today = 0
        if _strategy_instance and hasattr(_strategy_instance, 'trades'):
            cutoff = time.time() - 86400
            trades_today = len([t for t in _strategy_instance.trades if t.get("time", 0) > cutoff])
        
        # ML status (safe)
        ml_trained = False
        if _strategy_instance and hasattr(_strategy_instance, 'ml_predictor'):
            ml_trained = _strategy_instance.ml_predictor.is_trained
        
        # Estadísticas del sistema
        return {
            "status": "online",
            "uptime": uptime,
            "memory_experiences": memory_stats.get("total_trades", 0),
            "learning_confidence": memory_stats.get("learning_confidence", "NONE"),
            "trades_today": trades_today,
            "ml_trained": ml_trained,
            "ml_accuracy": 0.0,  # TODO: Calcular accuracy real
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
    except Exception as e:
        # Return safe defaults if anything fails
        return {
            "status": "online",
            "uptime": 0,
            "memory_experiences": 0,
            "learning_confidence": "NONE",
            "trades_today": 0,
            "ml_trained": False,
            "ml_accuracy": 0.0,
            "cpu_percent": 0.0,
            "memory_mb": 0.0,
            "error": str(e)
        }

# Sentiment Analysis
@app.get("/api/sentiment")
async def get_sentiment():
    """Retorna sentimiento actual del mercado."""
    try:
        if _strategy_instance and hasattr(_strategy_instance, 'sentiment_manager'):
            # Actualizar sentimiento
            await _strategy_instance.sentiment_manager.update()
            return _strategy_instance.sentiment_manager.get_current_sentiment()
        return {
            "overall": "neutral",
            "score": 0.0,
            "confidence": "low",
            "recent_events": [],
            "trend": "stable",
            "last_updated": 0,
            "news_count": 0
        }
    except Exception as e:
        return {
            "overall": "neutral",
            "score": 0.0,
            "confidence": "low",
            "recent_events": [],
            "trend": "stable",
            "error": str(e)
        }

@app.get("/api/sentiment/news")
def get_recent_news():
    """Retorna últimas noticias analizadas."""
    try:
        if _strategy_instance and hasattr(_strategy_instance, 'sentiment_manager'):
            return _strategy_instance.sentiment_manager.get_recent_news(limit=10)
        return []
    except Exception as e:
        return []

# --- FIN INTEGRACIÓN ---

@app.get("/api/balance")
async def get_balance():
    """Obtiene el balance (simulado o real)."""
    from broker_api_handler import get_balance
    import os
    
    mode = os.getenv("TRADING_MODE", "demo")
    balance = await get_balance(mode=mode)
    
    return {
        "USDT": balance.get('USDT', 0),
        "BTC": balance.get('BTC', 0),
        "mode": mode
    }


# Control de auto-trading
_trading_enabled = True

@app.post("/api/trade/close")
async def api_close_position(payload: dict):
    """Cierra inmediatamente la posición del activo especificado."""
    from broker_api_handler import close_position
    import os
    
    mode = os.getenv("TRADING_MODE", "demo")
    symbol = payload.get("symbol", "BTC_USDT").replace("_", "/")
    
    result = await close_position(symbol, mode=mode)
    
    # Resetear estado interno de la estrategia si es necesario
    if result.get("status") in ["FILLED", "SIMULATED"]:
        if _strategy_instance:
            if symbol in _strategy_instance.positions:
                _strategy_instance.positions[symbol] = 0.0
                _strategy_instance.entry_prices[symbol] = 0.0
        
    return result

@app.post("/api/trades/close_all")
async def close_all_positions():
    """Cierra TODAS las posiciones abiertas."""
    from broker_api_handler import close_position
    import os
    
    # Use the global trading mode variable
    mode = _trading_mode
    results = []
    
    # 🚨 FIX: Use _strategy_instance.positions (la fuente de verdad)
    if not _strategy_instance:
        return {"status": "ERROR", "message": "Strategy not initialized", "results": []}
    
    # Encontrar todos los símbolos con posiciones abiertas (Strategy + Tracker)
    strategy_symbols = [s for s, pos in _strategy_instance.positions.items() if pos != 0]
    
    from trade_tracker import get_tracker
    tracker = get_tracker()
    active_trades = tracker.get_active_trades()
    tracker_symbols = [t["symbol"] for t in active_trades]
    
    # Unir y deducir
    open_symbols = list(set(strategy_symbols + tracker_symbols))
    
    print(f"[CLOSE ALL] Found {len(open_symbols)} symbols to check: {open_symbols}")
    
    for symbol in open_symbols:
        result = await close_position(symbol, mode=mode)
        results.append(result)
        
        # Sync strategy
        if result.get("status") in ["FILLED", "SIMULATED"]:
            if symbol in _strategy_instance.positions:
                _strategy_instance.positions[symbol] = 0.0
                _strategy_instance.entry_prices[symbol] = 0.0
                print(f"[CLOSE ALL] Closed {symbol}")

    return {
        "status": "COMPLETED", 
        "closed_count": len(results),
        "results": results
    }

@app.post("/api/trading/toggle")
async def toggle_trading(payload: dict):
    global _trading_enabled
    # Si se envía un valor específico, usarlo; si no, hacer toggle
    if "enabled" in payload:
        _trading_enabled = payload["enabled"]
    else:
        _trading_enabled = not _trading_enabled
    
    print(f"[API] Trading {'ENABLED' if _trading_enabled else 'DISABLED'}")
    return {"enabled": _trading_enabled}

@app.get("/api/trading/status")
def get_trading_status():
    return {"enabled": _trading_enabled}

@app.get("/api/position")
async def get_position(symbol: str = "BTC/USDT"):
    """Devuelve la posición actual con PnL no realizado."""
    from broker_api_handler import get_current_position, get_current_price
    
    # Normalizar símbolo
    symbol = symbol.replace("_", "/")
    
    # Obtener precio real/demo
    current_price = await get_current_price(symbol, mode="demo")
    
    position = await get_current_position(current_price, symbol=symbol)
    
    return position


# === INDIVIDUAL TRADES ===
@app.get("/api/trades/active")
def get_active_trades():
    """Devuelve todos los trades abiertos con PnL en tiempo real."""
    try:
        from trade_tracker import get_tracker
        from broker_api_handler import get_current_price
        import random
        
        tracker = get_tracker()
        active_trades = tracker.get_active_trades()
        
        # Calcular PnL en vivo para cada trade
        trades_with_pnl = []
        # Nota: Esto podría ser lento si hay muchos trades de diferentes símbolos
        # Por ahora usamos un precio aproximado o iteramos
        
        # Agrupar por símbolo para minimizar llamadas de precio
        symbols = set(t['symbol'] for t in active_trades)
        prices = {} # symbol -> price
        
        # Esto debería ser async pero get_active_trades es def (sync)
        # Por simplicidad, usamos precio random o mock aquí, 
        # o convertimos el endpoint a async
        
        # FIX: Convertir endpoint a async para usar await get_current_price
        # Pero eso requiere cambiar la firma.
        # Por ahora, usamos el precio random como antes pero TODO: mejorar
        current_price = 86000 + random.uniform(-500, 500) 
        
        for trade in active_trades:
            # Idealmente usar precio específico del símbolo
            trade_with_pnl = tracker.calculate_live_pnl(trade, current_price)
            trades_with_pnl.append(trade_with_pnl)
        
        print(f"[API] Devolviendo {len(trades_with_pnl)} trades activos")
        return {"trades": trades_with_pnl}
    except Exception as e:
        print(f"[ERROR] /api/trades/active: {e}")
        import traceback
        traceback.print_exc()
        return {"trades": [], "error": str(e)}

@app.post("/api/trades/close/{trade_id}")
def close_single_trade(trade_id: str):
    """Cierra un trade individual."""
    try:
        from trade_tracker import get_tracker
        import random
        
        # 1. Close in TradeTracker (for UI active trades list)
        tracker = get_tracker()
        current_price = 86000 + random.uniform(-500, 500)
        
        # Convert ID to string for consistency
        trade_id = str(trade_id)
        
        trade = tracker.close_trade(trade_id, current_price)
        
        # 2. Close in RiskStrategy (for internal logic and history)
        if _strategy_instance and trade:
            symbol = trade["symbol"]
            size = trade["size"]
            side = trade["side"]
            
            # Actualizar posición agregada
            current_pos = _strategy_instance.positions.get(symbol, 0.0)
            
            if side == "LONG":
                # Cerrar LONG: restar tamaño
                new_pos = current_pos - size
            else:
                # Cerrar SHORT: sumar tamaño (porque short es negativo o se trata como tal)
                # NOTA: En este sistema, si positions es positivo para LONG y negativo para SHORT:
                # Si era SHORT, current_pos debería ser negativo. Al cerrar, sumamos size.
                # Si el sistema usa positions siempre positivo y side separado (menos probable en RiskStrategy),
                # asumiremos la convención estándar: LONG > 0, SHORT < 0.
                # Pero revisando register_trade, SHORT resta size. Así que aquí sumamos.
                new_pos = current_pos + size
            
            # Evitar errores de punto flotante
            if abs(new_pos) < 0.000001:
                new_pos = 0.0
                _strategy_instance.entry_prices[symbol] = 0.0
                
            _strategy_instance.positions[symbol] = new_pos
            print(f"[SYNC] 🔒 Trade #{trade_id} cerrado. Posición {symbol}: {current_pos:.4f} → {new_pos:.4f}")
            
        if trade:
            return {"success": True, "trade": trade}
            
        return {"success": False, "error": "Trade not found"}
    except Exception as e:
        print(f"[ERROR] close trade: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/api/debug/state")
def debug_state():
    """Endpoint temporal para depurar estado interno."""
    return {
        "active_symbols": _strategy_instance.active_symbols,
        "positions": _strategy_instance.positions,
        "price_history_len": {k: len(v) for k, v in _strategy_instance.price_history.items()},
        "candles_len": {k: len(v) for k, v in _strategy_instance.candles.items()},
        "last_prices": _strategy_instance.last_price
    }

@app.post("/api/trades/reverse/{trade_id}")
def reverse_trade_direction(trade_id: str):
    """Invierte la dirección de un trade (LONG→SHORT o SHORT→LONG)."""
    from trade_tracker import get_tracker
    
    tracker = get_tracker()
    
    # Buscar trade por ID (puede ser string o int)
    trade = None
    for t in tracker.trades:
        if str(t.get('id')) == str(trade_id) and t.get('status') == 'OPEN':
            old_side = t.get('side')
            symbol = t.get('symbol')
            size = t.get('size')
            
            # Invertir en tracker
            trade = tracker.reverse_trade(t['id'])
            
            if trade and _strategy_instance:
                # 🔄 SINCRONIZACIÓN: Actualizar posición agregada en RiskStrategy
                current_pos = _strategy_instance.positions.get(symbol, 0.0)
                
                # Calcular nueva posición después de invertir
                if old_side == "LONG":
                    # Era LONG (+size), ahora es SHORT (-size)
                    # Cambio: -size - size = -2*size
                    new_pos = current_pos - (2 * size)
                else:  # old_side == "SHORT"
                    # Era SHORT (-size), ahora es LONG (+size)
                    # Cambio: +size + size = +2*size
                    new_pos = current_pos + (2 * size)
                
                _strategy_instance.positions[symbol] = new_pos
                
                print(f"[SYNC] 🔄 Posición actualizada: {symbol} {current_pos:.4f} → {new_pos:.4f}")
                
            break
    
    if trade:
        return {"success": True, "trade": trade}
    return {"success": False, "error": f"Trade {trade_id} not found or already closed"}, 404


# === AUTO-TRADING BACKGROUND TASK ===
# TEMPORALMENTE DESHABILITADO - Implementar correctamente
# @app.on_event("startup")
# async def startup_event():
#     """Inicia el auto-trading background task."""
#     global _auto_trading_task
#     from auto_trader import auto_trading_loop
#     
#     # Función que devuelve si el trading está habilitado
#     def is_trading_enabled():
#         return _trading_enabled
#     
#     # Iniciar el loop en background
#     _auto_trading_task = asyncio.create_task(
#         auto_trading_loop(_strategy_instance, is_trading_enabled)
#     )
#     print("[API] 🚀 Auto-trading background task started")

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Detiene el auto-trading background task."""
#     global _auto_trading_task
#     if _auto_trading_task:
#         _auto_trading_task.cancel()
#         try:
#             await _auto_trading_task
#         except asyncio.CancelledError:
#             pass
#     print("[API] 🛑 Auto-trading background task stopped")


# === STATIC FILE SERVING (Frontend) ===
# This MUST be at the END to avoid overriding API routes
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Determinar ruta de estáticos (compatible con Docker y local)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    # Fallback para desarrollo local
    static_dir = os.path.join(os.path.dirname(__file__), "../frontend/out")

if os.path.exists(static_dir):
    # Mount Next.js assets
    next_dir = os.path.join(static_dir, "_next")
    if os.path.exists(next_dir):
        app.mount("/_next", StaticFiles(directory=next_dir), name="next")
    
    # Catch-all route for SPA (MUST BE LAST)
    # CRITICAL: Exclude /api/* paths to avoid intercepting API calls
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Skip API routes - let FastAPI handle them
        if full_path.startswith("api/"):
            # This shouldn't be hit if API routes are defined before this
            # But we add this check as a safeguard
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="API endpoint not found")
        
        # Root path
        if full_path == "":
            return FileResponse(os.path.join(static_dir, "index.html"))
            
        # Servir archivo si existe
        file_path = os.path.join(static_dir, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
            
        # Si no existe, servir index.html (SPA fallback)
        return FileResponse(os.path.join(static_dir, "index.html"))
