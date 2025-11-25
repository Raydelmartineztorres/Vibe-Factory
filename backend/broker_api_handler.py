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
# OANDA temporalmente deshabilitado (librería removida)
# from oandapyV20 import API
# from oandapyV20.endpoints.orders import OrderCreate

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

# Tracking de posición actual
_current_position = {
    "size": 0.0,           # Cantidad de BTC
    "entry_price": 0.0,    # Precio de entrada
    "is_open": False       # Si hay posición abierta
}

# Cache para la instancia del exchange
_exchange_instance = None
_oanda_client = None


async def _get_exchange(mode: Literal["testnet", "real", "coinbase"] = "real"):
    """
    Inicializa y retorna la instancia del exchange usando ccxt.
    Lee credenciales de variables de entorno según el modo.
    """
    global _exchange_instance
    
    # Si ya existe una instancia, verificar si es del modo correcto (esto es simplificado, 
    # idealmente deberíamos tener instancias separadas o reinicializar)
    # Por ahora, forzamos reinicialización si cambiamos de modo
    if _exchange_instance:
        # Check if current instance matches requested mode (heuristic)
        is_sandbox = _exchange_instance.urls['api'] == _exchange_instance.urls['test']
        if (mode == "testnet" and not is_sandbox) or (mode == "real" and is_sandbox):
            print(f"[BROKER] Switching exchange mode to {mode}...")
            await _exchange_instance.close()
            _exchange_instance = None
        else:
            return _exchange_instance

    exchange_id = os.getenv("EXCHANGE_ID", "binance")
    
    if mode == "coinbase":
        # Coinbase configuration
        exchange_id = "coinbase"
        api_key = os.getenv("COINBASE_API_KEY")
        secret = os.getenv("COINBASE_API_SECRET")
        print("[BROKER] Initializing COINBASE mode")
    elif mode == "testnet":
        api_key = os.getenv("BINANCE_TESTNET_KEY") or os.getenv("EXCHANGE_API_KEY")
        secret = os.getenv("BINANCE_TESTNET_SECRET") or os.getenv("EXCHANGE_SECRET")
        print("[BROKER] Initializing in TESTNET mode")
    else:
        api_key = os.getenv("BINANCE_REAL_KEY") or os.getenv("EXCHANGE_API_KEY")
        secret = os.getenv("BINANCE_REAL_SECRET") or os.getenv("EXCHANGE_SECRET")
        print("[BROKER] Initializing in REAL mode")

    if not api_key or not secret:
        raise ValueError(f"Faltan credenciales para modo {mode}. Configure BINANCE_{mode.upper()}_KEY y SECRET.")

    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange_config = {
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }

        _exchange_instance = exchange_class(exchange_config)
        
        # Activar modo Sandbox/Testnet
        if mode == "testnet":
            _exchange_instance.set_sandbox_mode(True)
        
        # Cargar mercados
        await _exchange_instance.load_markets()
        print(f"[BROKER] Conectado exitosamente a {exchange_id} ({mode})")
        return _exchange_instance
    except Exception as e:
        print(f"[ERROR] Fallo al conectar con {exchange_id}: {e}")
        raise e


# OANDA CLIENT DESHABILITADO TEMPORALMENTE
# def _get_oanda_client(mode: Literal["demo", "real"] = "demo"):
#     """
#     Inicializa y retorna cliente OANDA.
#     """
#     global _oanda_client
#     
#     if _oanda_client:
#         return _oanda_client
#     
#     if mode == "demo":
#         token = os.getenv("OANDA_DEMO_TOKEN")
#         account_id = os.getenv("OANDA_DEMO_ACCOUNT_ID")
#         environment = "practice"
#         print("[BROKER] Initializing OANDA DEMO mode")
#     else:
#         token = os.getenv("OANDA_REAL_TOKEN")
#         account_id = os.getenv("OANDA_REAL_ACCOUNT_ID")
#         environment = "live"
#         print("[BROKER] Initializing OANDA REAL mode")
#     
#     if not token or not account_id:
#         raise ValueError(f"Faltan credenciales OANDA para modo {mode}. Configure OANDA_{mode.upper()}_TOKEN y ACCOUNT_ID.")
#     
#     try:
#         _oanda_client = {"api": API(access_token=token, environment=environment), "account_id": account_id}
#         print(f"[BROKER] Conectado exitosamente a OANDA ({mode})")
#         return _oanda_client
#     except Exception as e:
#         print(f"[ERROR] Fallo al conectar con OANDA: {e}")
#         raise e


async def execute_order(payload: OrderPayload, mode: Literal["demo", "testnet", "real", "coinbase"]) -> dict:
    """
    Envía la orden al broker configurado o la simula.
    """
    global _simulated_balance, _trade_counter, _current_position
    
    symbol = payload["symbol"].replace("_", "/") # Convertir BTC_USDT a BTC/USDT
    side = payload["side"]
    size = payload["size"]
    
    # Extract base currency from symbol (e.g., BTC from BTC/USDT)
    base_currency = symbol.split("/")[0]

    # --- MODO DEMO (PAPER) ---
    if mode == "demo":
        # Precio simulado (usaremos uno realista)
        current_price = 90000 + random.uniform(-500, 500)
        
        # Initialize balance for this currency if it doesn't exist
        if base_currency not in _simulated_balance:
            _simulated_balance[base_currency] = 0.0
        
        # Ejecutar la operación en balance simulado
        if side == "BUY":
            cost = size * current_price
            if _simulated_balance["USDT"] >= cost:
                _simulated_balance["USDT"] -= cost
                _simulated_balance[base_currency] += size
                _trade_counter += 1
                
                # Actualizar posición
                _current_position["size"] = _simulated_balance[base_currency]
                _current_position["entry_price"] = current_price
                _current_position["is_open"] = True
                
                # Registrar en trade tracker
                from trade_tracker import get_tracker
                tracker = get_tracker()
                tracker.open_trade(size, current_price, "LONG", symbol=symbol)
                
                print(f"[SIMULATED] ✅ COMPRA ejecutada: {size} {base_currency} @ ${current_price:.2f}")
                return {
                    "status": "FILLED",
                    "id": f"SIM-{_trade_counter}",
                    "price": current_price,
                    "details": {"side": "BUY", "amount": size, "cost": cost, "price": current_price, "symbol": symbol}
                }
            else:
                return {"status": "FAILED", "error": f"Insufficient USDT balance (Req: ${cost:.2f}, Avail: ${_simulated_balance['USDT']:.2f})"}
                
        elif side == "SELL":
            if _simulated_balance.get(base_currency, 0) >= size:
                _simulated_balance[base_currency] -= size
                proceeds = size * current_price
                _simulated_balance["USDT"] += proceeds
                _trade_counter += 1
                
                # Actualizar posición
                _current_position["size"] = _simulated_balance[base_currency]
                _current_position["is_open"] = False # Simplificación: venta cierra posición
                
                # Registrar en trade tracker
                from trade_tracker import get_tracker
                tracker = get_tracker()
                tracker.close_trade(f"SIM-{_trade_counter}", current_price, proceeds - (size * _current_position["entry_price"])) # PnL aprox
                
                print(f"[SIMULATED] ✅ VENTA ejecutada: {size} {base_currency} @ ${current_price:.2f}")
                return {
                    "status": "FILLED",
                    "id": f"SIM-{_trade_counter}",
                    "price": current_price,
                    "details": {"side": "SELL", "amount": size, "proceeds": proceeds, "price": current_price, "symbol": symbol}
                }
            else:
                return {"status": "FAILED", "error": f"Insufficient {base_currency} balance"}

    # --- MODO REAL (Binance o Coinbase) ---
    elif mode in ["testnet", "real", "coinbase"]:
        try:
            exchange = await _get_exchange(mode)
            
            # Determinar el símbolo correcto para el exchange
            trading_symbol = symbol
            if mode == "coinbase":
                # Coinbase usa formato diferente (BTC-USD en vez de BTC/USDT)
                trading_symbol = symbol.replace("/USDT", "/USD").replace("/", "-")
            
            print(f"[{mode.upper()}] Ejecutando orden {side} de {size} {trading_symbol}")
            
            # Normalizar símbolo para ccxt
            market = exchange.market(trading_symbol)
            symbol_ccxt = market['symbol']
            
            # Ejecutar orden de mercado
            order = await exchange.create_order(
                symbol=symbol_ccxt,
                type='market',
                side=side.lower(),
                amount=size
            )
            
            print(f"[{mode.upper()}] ✅ Orden ejecutada: {order['id']}")
            return {
                "status": "FILLED",
                "id": order['id'],
                "price": order.get('price', 0) or order.get('average', 0),
                "details": order
            }
            
        except Exception as e:
            print(f"[{mode.upper()}] ❌ Error ejecutando orden: {e}")
            return {"status": "FAILED", "error": str(e)}
    
    # --- MODO DESCONOCIDO ---
    else:
        return {"status": "FAILED", "error": f"Modo '{mode}' no soportado"}


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
    global _simulated_balance, _trade_counter, _current_position
    
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
        
        # Resetear posición
        _current_position["size"] = 0.0
        _current_position["entry_price"] = 0.0
        _current_position["is_open"] = False
        
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

async def get_current_position(current_price: float) -> dict:
    """
    Obtiene la posición actual y calcula el PnL no realizado.
    
    Args:
        current_price: Precio actual del activo
        
    Returns:
        dict con: is_open, size, entry_price, current_price, unrealized_pnl_usd, unrealized_pnl_pct
    """
    global _current_position
    
    if not _current_position["is_open"] or _current_position["size"] <= 0:
        return {
            "is_open": False,
            "size": 0.0,
            "entry_price": 0.0,
            "current_price": current_price,
           "unrealized_pnl_usd": 0.0,
            "unrealized_pnl_pct": 0.0
        }
    
    # Calcular PnL no realizado
    entry_value = _current_position["size"] * _current_position["entry_price"]
    current_value = _current_position["size"] * current_price
    unrealized_pnl_usd = current_value - entry_value
    unrealized_pnl_pct = (unrealized_pnl_usd / entry_value) * 100 if entry_value > 0 else 0.0
    
    return {
        "is_open": True,
        "size": _current_position["size"],
        "entry_price": _current_position["entry_price"],
        "current_price": current_price,
        "unrealized_pnl_usd": unrealized_pnl_usd,
        "unrealized_pnl_pct": unrealized_pnl_pct
    }
