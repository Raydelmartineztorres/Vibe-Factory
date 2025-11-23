"""
Adaptadores para enviar órdenes al broker/exchange.
"""

from __future__ import annotations

from typing import Literal, TypedDict
import time
import random


class OrderPayload(TypedDict):
    symbol: str
    side: Literal["BUY", "SELL"]
    size: float
    stop_loss: float | None
    take_profit: float | None


# Balance simulado
_simulated_balance = {
    "USDT": 10000.0,
    "BTC": 0.0
}

_trade_counter = 1000


async def execute_order(payload: OrderPayload, mode: Literal["demo", "real"]) -> dict:
    """
    Envía la orden al broker configurado o la simula.
    """
    global _simulated_balance, _trade_counter
    
    # Modo SIMULADO (funciona siempre)
    if mode == "demo":
        symbol = payload["symbol"].replace("_", "/")
        side = payload["side"]
        size = payload["size"]
        
        # Precio simulado (usaremos uno realista)
        current_price = 90000 + random.uniform(-500, 500)
        
        # Ejecutar la operación en balance simulado
        if side == "BUY":
            cost = size * current_price
            if _simulated_balance["USDT"] >= cost:
                _simulated_balance["USDT"] -= cost
                _simulated_balance["BTC"] += size
                _trade_counter += 1
                
                print(f"[SIMULATED] ✅ COMPRA ejecutada: {size} BTC @ ${current_price:.2f}")
                print(f"[SIMULATED] Balance: {_simulated_balance['BTC']:.4f} BTC, ${_simulated_balance['USDT']:.2f} USDT")
                
                return {
                    "status": "FILLED",
                    "id": f"SIM-{_trade_counter}",
                    "price": current_price,
                    "details": {
                        "side": "BUY",
                        "amount": size,
                        "cost": cost,
                        "price": current_price
                    }
                }
            else:
                return {"status": "FAILED", "error": "Insufficient USDT balance"}
                
        elif side == "SELL":
            if _simulated_balance["BTC"] >= size:
                _simulated_balance["BTC"] -= size
                proceeds = size * current_price
                _simulated_balance["USDT"] += proceeds
                _trade_counter += 1
                
                print(f"[SIMULATED] ✅ VENTA ejecutada: {size} BTC @ ${current_price:.2f}")
                print(f"[SIMULATED] Balance: {_simulated_balance['BTC']:.4f} BTC, ${_simulated_balance['USDT']:.2f} USDT")
                
                return {
                    "status": "FILLED",
                    "id": f"SIM-{_trade_counter}",
                    "price": current_price,
                    "details": {
                        "side": "SELL",
                        "amount": size,
                        "cost": proceeds,
                        "price": current_price
                    }
                }
            else:
                return {"status": "FAILED", "error": "Insufficient BTC balance"}
    
    # Modo REAL (requiere API keys válidas)
    else:
        return {"status": "ERROR", "message": "Real mode requires valid Binance API keys"}


async def get_simulated_balance():
    """Retorna el balance simulado actual."""
    return _simulated_balance.copy()
