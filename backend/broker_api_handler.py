"""
Adaptadores para enviar órdenes al broker/exchange.
Soporta modo DEMO (simulado) y REAL (vía ccxt).
"""

from __future__ import annotations

from typing import Literal, TypedDict
import time
import random
import os
import ccxt.async_support as ccxt  # Async version of ccxt

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

# Cache para la instancia del exchange
_exchange_instance = None


async def _get_exchange():
    """
    Inicializa y retorna la instancia del exchange usando ccxt.
    Lee credenciales de variables de entorno.
    """
    global _exchange_instance
    if _exchange_instance:
        return _exchange_instance

    exchange_id = os.getenv("EXCHANGE_ID", "binance")  # Default to binance
    api_key = os.getenv("EXCHANGE_API_KEY")
    secret = os.getenv("EXCHANGE_SECRET")
    password = os.getenv("EXCHANGE_PASSWORD") # Required for some exchanges like KuCoin/OKX

    if not api_key or not secret:
        raise ValueError("Faltan credenciales: EXCHANGE_API_KEY y EXCHANGE_SECRET son requeridos para modo REAL.")

    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange_config = {
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        }
        if password:
            exchange_config['password'] = password

        _exchange_instance = exchange_class(exchange_config)
        
        # Activar modo Sandbox/Testnet si se requiere
        is_sandbox = os.getenv("EXCHANGE_SANDBOX", "false").lower() == "true"
        if is_sandbox:
            _exchange_instance.set_sandbox_mode(True)
            print(f"[REAL] 🧪 Modo Sandbox (Testnet) ACTIVADO para {exchange_id}")
        
        # Cargar mercados (necesario para normalizar símbolos)
        await _exchange_instance.load_markets()
        print(f"[REAL] Conectado exitosamente a {exchange_id}")
        return _exchange_instance
    except Exception as e:
        print(f"[ERROR] Fallo al conectar con {exchange_id}: {e}")
        raise e


async def execute_order(payload: OrderPayload, mode: Literal["demo", "real"]) -> dict:
    """
    Envía la orden al broker configurado o la simula.
    """
    global _simulated_balance, _trade_counter
    
    symbol = payload["symbol"].replace("_", "/") # Convertir BTC_USDT a BTC/USDT
    side = payload["side"]
    size = payload["size"]

    # --- MODO DEMO ---
    if mode == "demo":
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
                return {
                    "status": "FILLED",
                    "id": f"SIM-{_trade_counter}",
                    "price": current_price,
                    "details": {"side": "BUY", "amount": size, "cost": cost, "price": current_price}
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
                return {
                    "status": "FILLED",
                    "id": f"SIM-{_trade_counter}",
                    "price": current_price,
                    "details": {"side": "SELL", "amount": size, "cost": proceeds, "price": current_price}
                }
            else:
                return {"status": "FAILED", "error": "Insufficient BTC balance"}
    
    # --- MODO REAL ---
    else:
        try:
            exchange = await _get_exchange()
            
            # Crear orden de mercado
            # Nota: Algunos exchanges requieren 'create_market_buy_order' especifico, 
            # pero create_order es el método unificado.
            print(f"[REAL] Enviando orden {side} {size} {symbol} a {exchange.id}...")
            
            order = await exchange.create_order(
                symbol=symbol,
                type='market',
                side=side.lower(), # ccxt usa 'buy'/'sell' en minusculas
                amount=size
            )
            
            print(f"[REAL] ✅ Orden ejecutada: {order['id']}")
            return {
                "status": "FILLED",
                "id": str(order['id']),
                "price": order.get('price') or order.get('average') or 0.0, # Precio real de ejecución
                "details": order
            }
            
        except Exception as e:
            print(f"[REAL] ❌ Error ejecutando orden: {e}")
            return {"status": "ERROR", "message": str(e)}


async def get_balance(mode: Literal["demo", "real"] = "demo"):
    """Retorna el balance (simulado o real)."""
    if mode == "demo":
        return _simulated_balance.copy()
    else:
        try:
            exchange = await _get_exchange()
            balance = await exchange.fetch_balance()
            
            # Normalizar respuesta para el frontend (USDT y BTC)
            # ccxt retorna 'total' con los balances totales
            return {
                "USDT": balance['total'].get('USDT', 0.0),
                "BTC": balance['total'].get('BTC', 0.0)
            }
        except Exception as e:
            print(f"[REAL] Error obteniendo balance: {e}")
            return {"USDT": 0.0, "BTC": 0.0, "error": str(e)}

async def close_position(symbol: str, mode: Literal["demo", "real"]) -> dict:
    """
    Cierra inmediatamente toda la posición del activo base (vende todo a mercado).
    """
    global _simulated_balance, _trade_counter
    
    # Normalizar símbolo (ej: BTC_USDT -> BTC/USDT)
    normalized_symbol = symbol.replace("_", "/")
    base_asset = normalized_symbol.split("/")[0] # BTC
    
    if mode == "demo":
        amount = _simulated_balance.get(base_asset, 0.0)
        if amount <= 0:
            return {"status": "FAILED", "message": "No position to close"}
            
        # Simular venta de todo
        current_price = 90000 + random.uniform(-500, 500)
        proceeds = amount * current_price
        
        _simulated_balance[base_asset] = 0.0
        _simulated_balance["USDT"] += proceeds
        _trade_counter += 1
        
        print(f"[SIMULATED] 🚨 CLOSE POSITION: Sold {amount} {base_asset} @ ${current_price:.2f}")
        return {
            "status": "FILLED",
            "id": f"SIM-CLOSE-{_trade_counter}",
            "price": current_price,
            "details": {"side": "SELL", "amount": amount, "cost": proceeds, "type": "CLOSE_POSITION"}
        }
        
    else:
        # Modo REAL
        try:
            exchange = await _get_exchange()
            
            # 1. Obtener balance del activo base
            balance = await exchange.fetch_balance()
            amount = balance['total'].get(base_asset, 0.0)
            
            # Verificar si hay suficiente para vender (evitar dust)
            # Esto es una simplificación, idealmente verificar min_amount del mercado
            if amount <= 0.00001: 
                return {"status": "FAILED", "message": f"Insufficient {base_asset} balance to close ({amount})"}
                
            print(f"[REAL] 🚨 CERRANDO POSICIÓN: Vendiendo {amount} {base_asset} en {exchange.id}...")
            
            # 2. Enviar orden de venta de mercado por el total
            order = await exchange.create_order(
                symbol=normalized_symbol,
                type='market',
                side='sell',
                amount=amount
            )
            
            print(f"[REAL] ✅ Posición cerrada: {order['id']}")
            return {
                "status": "FILLED",
                "id": str(order['id']),
                "price": order.get('price') or order.get('average') or 0.0,
                "details": order
            }
            
        except Exception as e:
            print(f"[REAL] ❌ Error cerrando posición: {e}")
            return {"status": "ERROR", "message": str(e)}

async def close_connection():
    """Cierra la conexión con el exchange si existe."""
    global _exchange_instance
    if _exchange_instance:
        await _exchange_instance.close()
        _exchange_instance = None
