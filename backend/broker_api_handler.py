"""
Broker API Handler (CCXT-based)
Handles trades for:
- Demo (paper trading simulation)
- Binance Testnet
- Binance Real
- Coinbase
"""

from typing import Literal, TypedDict
import os
import random
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

# Tracking de posiciones por símbolo
_positions = {} # {"BTC/USDT": {"size": 0.5, "entry_price": 90000, "is_open": True}}

# Cache para la instancia del exchange
_exchange_instance = None


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
        
        if not api_key or not secret:
            raise ValueError("Faltan credenciales para Coinbase. Configure COINBASE_API_KEY y COINBASE_API_SECRET en Render.")
        
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


async def execute_order(payload: OrderPayload, mode: Literal["demo", "testnet", "real", "coinbase"]) -> dict:
    """
    Envía la orden al broker configurado o la simula.
    """
    global _simulated_balance, _trade_counter, _positions
    
    symbol = payload["symbol"].replace("_", "/") # Convertir BTC_USDT a BTC/USDT
    side = payload["side"]
    size = payload["size"]
    
    # Extract base currency from symbol (e.g., BTC from BTC/USDT)
    base_currency = symbol.split("/")[0]

    # --- MODO DEMO (PAPER) ---
    if mode == "demo":
        # Precio simulado (usaremos uno realista)
        # FIX: Usar precio real del símbolo, no hardcodeado a BTC
        current_price = await get_current_price(symbol, mode="demo")
        
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
                
                # Actualizar posición para este símbolo
                if symbol not in _positions:
                    _positions[symbol] = {"size": 0.0, "entry_price": 0.0, "is_open": False}
                
                # Calcular nuevo precio promedio si ya hay posición
                current_pos = _positions[symbol]
                if current_pos["size"] > 0:
                    total_cost = (current_pos["size"] * current_pos["entry_price"]) + cost
                    new_size = current_pos["size"] + size
                    _positions[symbol]["entry_price"] = total_cost / new_size
                    _positions[symbol]["size"] = new_size
                else:
                    _positions[symbol]["entry_price"] = current_price
                    _positions[symbol]["size"] = size
                
                _positions[symbol]["is_open"] = True
                
                _positions[symbol]["is_open"] = True
                
                # Registrar en trade tracker - MOVED TO RISK STRATEGY
                # from trade_tracker import get_tracker
                # tracker = get_tracker()
                
                # Use consistent ID format
                trade_id = f"SIM-{_trade_counter}"
                
                # Open trade in tracker - REMOVED
                # tracker.open_trade(size, current_price, "LONG", symbol=symbol, trade_id=trade_id)
                
                print(f"[SIMULATED] ✅ COMPRA ejecutada: {size} {base_currency} @ ${current_price:.2f}")
                return {
                    "status": "FILLED",
                    "id": trade_id,
                    "price": current_price,
                    "details": {"side": "BUY", "amount": size, "cost": cost, "price": current_price, "symbol": symbol}
                }
            else:
                return {"status": "FAILED", "error": f"Insufficient USDT balance (Req: ${cost:.2f}, Avail: ${_simulated_balance['USDT']:.2f})"}
                
        elif side == "SELL":
            # En demo, permitimos "shorting" (balance negativo de BTC)
            # if _simulated_balance.get(base_currency, 0) >= size: <--- Removed check
            
            # Inicializar si no existe
            if base_currency not in _simulated_balance:
                _simulated_balance[base_currency] = 0.0
                
            _simulated_balance[base_currency] -= size
            proceeds = size * current_price
            _simulated_balance["USDT"] += proceeds
            _trade_counter += 1
            
            # Actualizar posición
            if symbol not in _positions:
                _positions[symbol] = {"size": 0.0, "entry_price": 0.0, "is_open": False}
                
            # Reducir posición (FIFO simplificado)
            current_pos = _positions[symbol]
            new_size = current_pos["size"] - size
            if new_size <= 0.000001: # Close if negligible
                new_size = 0.0
                current_pos["is_open"] = False
                current_pos["entry_price"] = 0.0
            
            current_pos["size"] = new_size
            _positions[symbol] = current_pos
            
            _positions[symbol] = current_pos
            
            # Registrar en trade tracker - MOVED TO RISK STRATEGY
            # from trade_tracker import get_tracker
            # tracker = get_tracker()
            
            # Use consistent ID format
            trade_id = f"SIM-{_trade_counter}"
            
            # Para short, el PnL se calcula al cerrar, aquí solo abrimos - REMOVED
            # tracker.open_trade(size, current_price, "SHORT", symbol=symbol, trade_id=trade_id)
            
            print(f"[SIMULATED] ✅ VENTA (SHORT) ejecutada: {size} {base_currency} @ ${current_price:.2f}")
            return {
                "status": "FILLED",
                "id": trade_id,
                "price": current_price,
                "details": {"side": "SELL", "amount": size, "proceeds": proceeds, "price": current_price, "symbol": symbol}
            }

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


async def get_current_price(symbol: str, mode: Literal["demo", "real"] = "demo") -> float:
    """
    Obtiene el precio actual del símbolo.
    
    IMPORTANTE: En modo demo, obtiene precios REALES de Binance (sin autenticación)
    para que EL GATO aprenda con datos reales del mercado.
    Solo las operaciones son simuladas, los precios son reales.
    """
    try:
        # Usar Binance público para obtener precios reales (funciona en demo y real)
        import ccxt.async_support as ccxt
        
        # Crear exchange sin credenciales (API pública)
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        
        # Obtener ticker actual
        ticker = await exchange.fetch_ticker(symbol)
        await exchange.close()
        
        price = float(ticker['last'])
        
        if mode == "demo":
            print(f"[DEMO-REAL-PRICE] 📊 {symbol}: ${price:.2f} (Binance Live)")
        
        return price
        
    except Exception as e:
        print(f"[PRICE] ⚠️ Error fetching from Binance: {e}")
        
        # Fallback: usar precio base realista si falla la conexión
        # (Evita que el bot se detenga por problemas de red)
        import time
        base_price = 87500.0
        time_factor = time.time() % 3600
        trend = 500 * ((time_factor / 1800) - 1)
        volatility = random.uniform(-400, 400)
        fallback_price = base_price + trend + volatility
        
        print(f"[PRICE] 🔄 Usando fallback: ${fallback_price:.2f}")
        return fallback_price

async def get_balance(mode: Literal["demo", "real"] = "demo"):
    """Retorna el balance (simulado o real)."""
    if mode == "demo":
        return _simulated_balance.copy()
    else:
        try:
            exchange = await _get_exchange(mode=mode)
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
    global _simulated_balance, _trade_counter, _positions
    
    # Normalizar símbolo (ej: BTC_USDT -> BTC/USDT)
    normalized_symbol = symbol.replace("_", "/")
    base_asset = normalized_symbol.split("/")[0] # BTC
    
    if mode == "demo":
        amount = _simulated_balance.get(base_asset, 0.0)
        
        # Sync with TradeTracker FIRST to handle ghost trades
        from trade_tracker import get_tracker
        tracker = get_tracker()
        active_trades = tracker.get_active_trades()
        closed_ghosts = 0
        
        # We use a current price for closing
        # FIX: Usar precio real del símbolo
        current_price = await get_current_price(normalized_symbol, mode="demo")
        
        for trade in active_trades:
            if trade["symbol"] == normalized_symbol:
                tracker.close_trade(trade["id"], current_price)
                closed_ghosts += 1
                print(f"[SIMULATED] 🔄 Tracker synced: Closed trade #{trade['id']}")

        if amount <= 0:
            if closed_ghosts > 0:
                return {
                    "status": "FILLED",
                    "id": f"SIM-GHOST-CLOSE-{_trade_counter}",
                    "price": current_price,
                    "details": {"side": "SELL", "amount": 0, "cost": 0, "type": "CLOSE_GHOST_TRADES", "closed_count": closed_ghosts}
                }
            return {"status": "FAILED", "message": "No position to close"}
            
        # Simular venta de todo
        proceeds = amount * current_price
        
        _simulated_balance[base_asset] = 0.0
        _simulated_balance["USDT"] += proceeds
        _trade_counter += 1
        
        # Resetear posición
        if normalized_symbol in _positions:
            _positions[normalized_symbol] = {"size": 0.0, "entry_price": 0.0, "is_open": False}
        
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
            exchange = await _get_exchange(mode=mode)
            
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
            # Sync with TradeTracker
            from trade_tracker import get_tracker
            tracker = get_tracker()
            active_trades = tracker.get_active_trades()
            exit_price = order.get('price') or order.get('average') or 0.0
            
            for trade in active_trades:
                if trade["symbol"] == normalized_symbol:
                    tracker.close_trade(trade["id"], exit_price)
                    print(f"[REAL] 🔄 Tracker synced: Closed trade #{trade['id']}")

            return {
                "status": "FILLED",
                "id": str(order['id']),
                "price": exit_price,
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

async def get_current_position(current_price: float, symbol: str = "BTC/USDT") -> dict:
    """
    Obtiene la posición actual y calcula el PnL no realizado.
    
    Args:
        current_price: Precio actual del activo
        symbol: Símbolo a consultar
        
    Returns:
        dict con: is_open, size, entry_price, current_price, unrealized_pnl_usd, unrealized_pnl_pct
    """
    global _positions
    
    position = _positions.get(symbol, {"size": 0.0, "entry_price": 0.0, "is_open": False})
    
    if not position["is_open"] or position["size"] == 0:
        return {
            "is_open": False,
            "size": 0.0,
            "entry_price": 0.0,
            "current_price": current_price,
            "unrealized_pnl_usd": 0.0,
            "unrealized_pnl_pct": 0.0
        }
    
    # Calcular PnL no realizado
    entry_value = position["size"] * position["entry_price"]
    current_value = position["size"] * current_price
    
    unrealized_pnl_usd = current_value - entry_value
    unrealized_pnl_pct = (unrealized_pnl_usd / abs(entry_value)) * 100 if entry_value != 0 else 0.0
    
    return {
        "is_open": True,
        "size": position["size"],
        "entry_price": position["entry_price"],
        "current_price": current_price,
        "unrealized_pnl_usd": unrealized_pnl_usd,
        "unrealized_pnl_pct": unrealized_pnl_pct
    }
