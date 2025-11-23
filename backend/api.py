from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Vibe Factory API")

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
async def api_trade(payload: dict):
    from broker_api_handler import execute_order
    # Default to demo mode for now
    result = await execute_order(payload, mode="demo")
    
    # Actualizar estado de la estrategia si la orden fue exitosa
    if result.get("status") in ["FILLED", "SIMULATED"]:
        _strategy_instance.register_trade(
            side=payload["side"],
            price=result["price"],
            size=payload["size"],
            result=result
        )
        
    return result


# --- INTEGRACIÓN DE ESTRATEGIA ---
from risk_strategy import RiskStrategy
from data_collector import bootstrap_data_pipeline
from optimizer import Optimizer
import asyncio
import time

_strategy_instance = RiskStrategy()
_optimizer_instance = Optimizer(_strategy_instance)

@app.on_event("startup")
async def startup_event():
    """Iniciar el loop de trading y aprendizaje en background."""
    # 1. Pipeline de datos y trading
    asyncio.create_task(bootstrap_data_pipeline(
        strategy=_strategy_instance,
        live_mode="demo"
    ))
    
    # 2. Optimizador (Aprendizaje)
    asyncio.create_task(_optimizer_instance.start_loop())

@app.get("/api/price")
def get_current_price():
    """Retorna el precio real de la estrategia."""
    price = _strategy_instance.last_price if _strategy_instance.last_price > 0 else 90000.0
    return {"symbol": "BTC/USDT", "price": price}

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
    candles = [
        {
            "time": c.time,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close
        }
        for c in _strategy_instance.candles
    ]
    # Agregar vela actual en formación
    if _strategy_instance.current_candle:
        c = _strategy_instance.current_candle
        candles.append({
            "time": c.time,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close
        })
    return {"candles": candles}

    return {"candles": candles}

@app.get("/api/memory")
def get_memory_stats():
    """Retorna estadísticas del módulo de memoria."""
    return _strategy_instance.memory.get_stats()

# --- FIN INTEGRACIÓN ---

@app.get("/api/balance")
async def get_balance():
    """Obtiene el balance simulado."""
    from broker_api_handler import get_simulated_balance
    balance = await get_simulated_balance()
    return {
        "USDT": balance.get('USDT', 0),
        "BTC": balance.get('BTC', 0),
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
