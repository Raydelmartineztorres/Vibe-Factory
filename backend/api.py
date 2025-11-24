from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import asyncio

app = FastAPI(title="Vibe Factory API")

# Server start time for uptime tracking
_server_start_time = time.time()

# Auto-trading background task
_auto_trading_task = None

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
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
app.add_middleware(BasicAuthMiddleware)

@app.get("/api/health")
def health_check():
    return {"status": "online", "message": "Vibe Factory Backend is running"}


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
        _strategy_instance.register_trade(
            side=payload["side"],
            price=result["price"],
            size=payload["size"],
            result=result
        )
        
    return result

@app.post("/api/trade/close")
async def api_close_position(payload: dict):
    """Cierra inmediatamente la posición del activo especificado."""
    from broker_api_handler import close_position
    import os
    
    mode = os.getenv("TRADING_MODE", "demo")
    symbol = payload.get("symbol", "BTC_USDT")
    
    result = await close_position(symbol, mode=mode)
    
    # Resetear estado interno de la estrategia si es necesario
    if result.get("status") in ["FILLED", "SIMULATED"]:
        _strategy_instance.position = 0.0
        
    return result


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
    global _data_task
    # 1. Pipeline de datos y trading
    _data_task = asyncio.create_task(bootstrap_data_pipeline(
        strategy=_strategy_instance,
        live_mode="demo", # Data pipeline always runs in demo/paper mode for now
        symbol=_current_symbol
    ))
    
    # 2. Optimizador (Aprendizaje)
    asyncio.create_task(_optimizer_instance.start_loop())

@app.get("/api/price")
def get_current_price():
    """Retorna el precio real de la estrategia."""
    price = _strategy_instance.last_price if _strategy_instance.last_price > 0 else 90000.0
    return {"symbol": _current_symbol, "price": price}

@app.post("/api/set_symbol")
async def set_symbol(payload: dict):
    """Cambia el símbolo activo y reinicia el pipeline de datos."""
    global _current_symbol, _data_task
    
    new_symbol = payload.get("symbol")
    if not new_symbol:
        return {"error": "Symbol required"}
        
    if new_symbol == _current_symbol:
        return {"status": "unchanged", "symbol": _current_symbol}
        
    print(f"[API] Switching symbol to {new_symbol}...")
    _current_symbol = new_symbol
    
    # Cancelar tarea anterior
    if _data_task:
        _data_task.cancel()
        try:
            await _data_task
        except asyncio.CancelledError:
            pass
            
    # Reiniciar pipeline con nuevo símbolo
    _data_task = asyncio.create_task(bootstrap_data_pipeline(
        strategy=_strategy_instance,
        live_mode="demo",
        symbol=_current_symbol
    ))
    
    # Resetear estado de la estrategia para el nuevo símbolo
    # (Opcional: podríamos querer mantener el historial si es multi-asset real)
    _strategy_instance.candles = []
    _strategy_instance.last_price = 0.0
    
    return {"status": "success", "symbol": _current_symbol}

@app.get("/api/trading/mode")
def get_trading_mode():
    """Retorna el modo de trading actual."""
    return {"mode": _trading_mode}

@app.post("/api/trading/mode")
def set_trading_mode(payload: dict):
    """Cambia el modo de trading (demo, testnet, real)."""
    global _trading_mode
    new_mode = payload.get("mode")
    if new_mode not in ["demo", "testnet", "real"]:
        return {"error": "Invalid mode. Use: demo, testnet, real"}
    
    _trading_mode = new_mode
    print(f"[API] Trading mode switched to: {_trading_mode.upper()}")
    return {"status": "success", "mode": _trading_mode}

@app.get("/api/trades")
def get_trades():
    """Retorna los trades ejecutados por la estrategia."""
    return {"trades": _strategy_instance.trades[-20:]}  # Últimos 20 trades

@app.get("/api/pnl")
def get_pnl():
    """Retorna métricas de PnL en tiempo real."""
    return {
        "realized_pnl": _strategy_instance.realized_pnl,
        "unrealized_pnl": _strategy_instance.unrealized_pnl,
        "position_size": _strategy_instance.position,
        "entry_price": _strategy_instance.entry_price,
        "current_price": _strategy_instance.last_price
    }

@app.get("/api/candles")
def get_candles():
    """Retorna las velas OHLC para el gráfico."""
    """Get candlestick data for the chart."""
    if _strategy_instance is None:
        return {"candles": []}
    
    candles = _strategy_instance.candles
    if _strategy_instance.current_candle:
        candles_to_send = candles + [_strategy_instance.current_candle]
    else:
        candles_to_send = candles
    
    return {
        "candles": [
            {
                "time": int(c.time),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
            }
            for c in candles_to_send
        ]
    }


@app.get("/api/indicators")
def get_indicators():
    """Get RSI and ATR indicator values for chart overlay."""
    if _strategy_instance is None:
        return {"rsi": [], "atr": []}
    
    candles = _strategy_instance.candles
    if not candles or len(candles) < 14:
        return {"rsi": [], "atr": []}
    
    # Calculate RSI and ATR for each candle
    rsi_data = []
    atr_data = []
    
    for i in range(14, len(candles)):
        window = candles[i-14:i+1]
        closes = [c.close for c in window]
        
        # RSI calculation
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
        
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_data.append({"time": int(candles[i].time), "value": round(rsi, 2)})
        
        # ATR calculation (simplified)
        highs = [c.high for c in window]
        lows = [c.low for c in window]
        tr_values = [highs[j] - lows[j] for j in range(len(highs))]
        atr = sum(tr_values) / len(tr_values)
        
        atr_data.append({"time": int(candles[i].time), "value": round(atr, 2)})
    
    # Apply light smoothing (3-period SMA) to reduce visual noise
    def smooth_data(data, period=3):
        """Apply simple moving average for smoother lines."""
        if len(data) < period:
            return data
        smoothed = []
        for i in range(len(data)):
            if i < period - 1:
                smoothed.append(data[i])  # Keep first values as-is
            else:
                # Average the last 'period' values
                window_values = [data[j]["value"] for j in range(i - period + 1, i + 1)]
                avg_value = sum(window_values) / period
                smoothed.append({"time": data[i]["time"], "value": round(avg_value, 2)})
        return smoothed
    
    # Smooth both indicators
    rsi_data = smooth_data(rsi_data, period=3)
    atr_data = smooth_data(atr_data, period=3)
    
    # Calculate MACD (12, 26, 9) for all candles
    macd_data = []
    if len(candles) >= 26:
        for i in range(26, len(candles)):
            window = candles[i-26:i+1]
            closes = [c.close for c in window]
            
            # EMA 12
            ema_12_values = [closes[0]]
            multiplier_12 = 2 / (12 + 1)
            for close in closes[1:]:
                ema_12_values.append((close - ema_12_values[-1]) * multiplier_12 + ema_12_values[-1])
            
            # EMA 26
            ema_26 = sum(closes) / len(closes)  # Start with SMA
            for close in closes:
                ema_26 = (close - ema_26) * (2 / (26 + 1)) + ema_26
            
            # MACD Line
            ema_12 = ema_12_values[-1]
            macd_line = ema_12 - ema_26
            
            # For simplicity, use macd_line as histogram (signal would require 9 more periods)
            # In a real implementation, you'd calculate signal line and then histogram
            macd_data.append({
                "time": int(candles[i].time),
                "histogram": round(macd_line, 2)
            })
    
    return {
        "rsi": rsi_data,
        "atr": atr_data,
        "macd": macd_data
    }

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
    
    signal = "NEUTRAL"
    if prediction:
        current_price = _strategy_instance.last_price
        if prediction > current_price * 1.001:
            signal = "BUY"
        elif prediction < current_price * 0.999:
            signal = "SELL"
            
    return {
        "signal": signal,
        "predicted_price": prediction,
        "current_price": _strategy_instance.last_price,
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

@app.post("/api/trading/toggle")
def toggle_trading():
    global _trading_enabled
    _trading_enabled = not _trading_enabled
    return {"enabled": _trading_enabled}

@app.get("/api/trading/status")
def get_trading_status():
    return {"enabled": _trading_enabled}

@app.get("/api/position")
async def get_position():
    """Devuelve la posición actual con PnL no realizado."""
    from broker_api_handler import get_current_position
    import random
    
    current_price = 86000 + random.uniform(-500, 500)
    position = await get_current_position(current_price)
    
    return position


# === INDIVIDUAL TRADES ===
@app.get("/api/trades/active")
def get_active_trades():
    """Devuelve todos los trades abiertos con PnL en tiempo real."""
    try:
        from trade_tracker import get_tracker
        import random
        
        tracker = get_tracker()
        # Usar precio simulado (cambiar luego por precio real del gráfico)
        current_price = 86000 + random.uniform(-500, 500)
        
        active_trades = tracker.get_active_trades()
        
        # Calcular PnL en vivo para cada trade
        trades_with_pnl = []
        for trade in active_trades:
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
def close_single_trade(trade_id: int):
    """Cierra un trade individual."""
    try:
        from trade_tracker import get_tracker
        import random
        
        tracker = get_tracker()
        current_price = 86000 + random.uniform(-500, 500)
        
        trade = tracker.close_trade(trade_id, current_price)
        
        if trade:
            return {"success": True, "trade": trade}
        return {"success": False, "error": "Trade not found"}
    except Exception as e:
        print(f"[ERROR] close trade: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/trades/reverse/{trade_id}")
def reverse_trade_direction(trade_id: int):
    """Invierte la dirección de un trade (LONG→SHORT o SHORT→LONG)."""
    from trade_tracker import get_tracker
    
    tracker = get_tracker()
    trade = tracker.reverse_trade(trade_id)
    
    if trade:
        return {"success": True, "trade": trade}
    return {"success": False, "error": "Trade not found"}


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
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Root path
        if full_path == "":
            return FileResponse(os.path.join(static_dir, "index.html"))
            
        # Servir archivo si existe
        file_path = os.path.join(static_dir, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
            
        # Si no existe, servir index.html (SPA fallback)
        return FileResponse(os.path.join(static_dir, "index.html"))
