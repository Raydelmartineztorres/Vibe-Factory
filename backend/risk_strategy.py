"""
Motor de gestión de riesgo y position sizing con auto-trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, Protocol
from datetime import datetime
import random
import pandas as pd

from memory import MarketContext, TradeMemory
from strategy_profiles import get_strategy_manager
from trade_tracker import TradeTracker


@dataclass
class RiskConfig:
    """Config global de riesgo."""

    capital_virtual: float = 100_000.0
    risk_per_trade: float = 0.015  # 1.5 %
    stop_loss_pct: float = 0.04
    take_profit_pct: float = 0.08


@dataclass
class Candle:
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

class RiskStrategy:
    """Implementa los cálculos clave de exposición y auto-trading con ML e inteligencia adaptativa."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        
        # 🔥 MULTI-ASSET SUPPORT: Dictionary-based position tracking
        self.positions = {}  # {"BTC/USDT": 0.5, "ETH/USDT": -2.0}
        self.entry_prices = {}  # {"BTC/USDT": 86000, "ETH/USDT": 3200}
        self.price_history = {}  # {"BTC/USDT": [86000, 86100, ...], "ETH/USDT": [...]}
        self.candles = {}  # {"BTC/USDT": [Candle(...), ...], "ETH/USDT": [...]}
        self.current_candle = {}  # {"BTC/USDT": Candle(...), "ETH/USDT": Candle(...)}
        self.last_price = {}  # {"BTC/USDT": 86500, "ETH/USDT": 3250}
        self.last_trade_time = {}  # {"BTC/USDT": timestamp, "ETH/USDT": timestamp}
        self.unrealized_pnl_per_symbol = {}  # {"BTC/USDT": 0.0, ...}
        self.position_ids = {}  # {"BTC/USDT": "pos_1", ...}
        
        # Portfolio-level tracking
        self.realized_pnl = 0.0  # Total across all symbols
        self.unrealized_pnl = 0.0  # Total across all symbols  
        self.trades = []  # All trades with symbol tag
        self.position_id = 0
        
        # Active symbols (will be updated by volatility scanner)
        self.active_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]  # Start with top assets
        
        # 💰 PROFIT PROTECTION (Crecimiento Compuesto)
        import time
        self.daily_start_time = time.time()
        self.daily_realized_pnl = 0.0
        self.profit_lock_active = False
        self.circuit_breaker_active = False
        self.take_profit_pct = 2.0  # Close trades at +2% gain
        self.daily_lock_threshold = 1.0  # Stop at 100% of daily target (conservative)
        
        # 🛡️ RISK MANAGEMENT: Pending Trades Protection (prevent opposing positions)
        self.pending_trades = {}  # {symbol: {"side": "BUY"/"SELL", "timestamp": time}}
        
        # 🧠 INTELIGENCIA ADAPTATIVA
        from ml_predictor import MLPredictor
        from market_timing import MarketTiming
        from sentiment_manager import SentimentManager
        from risk_manager import RiskManager
        from volatility_scanner import VolatilityScanner
        from el_gato import ElGato
        import asyncio
        
        # ML y Memory config (por defecto activados)
        self.ml_enabled = False
        self.memory_enabled = True
        
        # Inicializar componentes
        self.ml_predictor = MLPredictor()
        self.market_timing = MarketTiming()
        self.sentiment_manager = SentimentManager()
        self.risk_manager = RiskManager(initial_capital=self.config.capital_virtual)
        self.memory = TradeMemory()
        self.volatility_scanner = VolatilityScanner()
        self.el_gato = ElGato()
        self.last_prediction = None
        
        # Scanner loop will be started by the event loop in api.py startup
        # asyncio.create_task(self._run_scanner_loop())  # Moved to api.py startup
        
        # Sistema de estrategias múltiples
        self.strategy_manager = get_strategy_manager()
        
        # 🆕 Trade Tracker for persistent storage
        from trade_tracker import get_tracker
        self.trade_tracker = get_tracker()
        
        # 🧹 Limpiar duplicados al inicio
        self.trade_tracker.cleanup_duplicates()
        
        # 🔮 PREDICTOR DE VELAS: Predice siguiente vela con 99% precisión
        from candle_predictor import CandlePredictor
        self.candle_predictor = CandlePredictor(lookback_period=30)
        self.predictions = {}  # {symbol: PredictedCandle}
        
        print(f"[STRATEGY] ✅ Inicializado con estrategia: {self.strategy_manager.active_strategy}")
        
        # 🔄 SYNC POSITIONS from TradeTracker (Recover state after restart)
        print(f"[STRATEGY] 🔄 Syncing positions from TradeTracker...")
        restored_count = 0
        
        # 🚨 SIMULACIÓN DE MERCADO REAL (FEES + SLIPPAGE)
        # Esto es CRÍTICO para tener resultados realistas
        self.SIMULATE_REAL_MARKET = True
        self.MAKER_FEE = 0.001  # 0.1% (Binance Standard)
        self.TAKER_FEE = 0.001  # 0.1% (Binance Standard)
        self.SLIPPAGE_MEAN = 0.0002  # 0.02% promedio
        self.accumulated_fees = 0.0
        
        print(f"[REALITY CHECK] ⚠️ Simulación de Mercado Real ACTIVADA")
        print(f"  └─ Fees: {self.TAKER_FEE*100}% por trade")
        print(f"  └─ Slippage estimado: ~{self.SLIPPAGE_MEAN*100}%")
        for trade in self.trade_tracker.trades:
            if trade.get("status") == "OPEN":
                symbol = trade["symbol"]
                size = trade["size"]
                side = trade["side"]
                entry_price = trade["entry_price"]
                
                # Determine sign based on side
                signed_size = size if side == "LONG" else -size
                
                # Update internal state
                self.positions[symbol] = signed_size
                self.entry_prices[symbol] = entry_price
                
                # Restore position ID
                self.position_id += 1
                self.position_ids[symbol] = f"pos_{self.position_id}"
                
                # Ensure symbol is in active list
                if symbol not in self.active_symbols:
                    self.active_symbols.append(symbol)
                
                # Initialize symbol tracking if needed
                self._init_symbol(symbol)
                
                restored_count += 1
                print(f"[STRATEGY] ♻️ Restored position: {symbol} {side} {size} @ ${entry_price}")
        
        if restored_count > 0:
            print(f"[STRATEGY] ✅ Restored {restored_count} active positions.")
        else:
            print(f"[STRATEGY] ℹ️ No active positions found to restore.")

    async def _run_scanner_loop(self):
        """Ejecuta el scanner de volatilidad periódicamente."""
        import asyncio
        while True:
            try:
                print("[SCANNER] 🔍 Iniciando escaneo de volatilidad...")
                top_assets = await self.volatility_scanner.scan_volatility()
                
                if top_assets:
                    new_symbols = [asset['symbol'] for asset in top_assets]
                    print(f"[SCANNER] ✅ Nuevos activos seleccionados: {new_symbols}")
                    
                    # Actualizar lista de activos activos
                    # Mantener activos donde ya tenemos posición abierta
                    current_positions = [s for s, pos in self.positions.items() if pos != 0]
                    self.active_symbols = list(set(new_symbols + current_positions))
                    
                    # Inicializar nuevos símbolos
                    for symbol in self.active_symbols:
                        self._init_symbol(symbol)
                else:
                    print("[SCANNER] ⚠️ No se encontraron activos volátiles, manteniendo actuales.")
                    
            except Exception as e:
                print(f"[SCANNER] ❌ Error en ciclo de escaneo: {e}")
                
            # Esperar 5 minutos antes del próximo escaneo
            await asyncio.sleep(300)

    async def execute_strategy(self):
        """
        Punto de entrada para el auto-trader.
        Obtiene el precio actual y ejecuta el ciclo de análisis (on_tick) para CADA activo.
        """
        from broker_api_handler import get_current_price
        
        results = []
        
        # Iterar sobre todos los símbolos activos
        # Copiamos la lista para evitar problemas si se modifica durante la iteración
        symbols_to_trade = list(self.active_symbols)
        if "BTC/USDT" not in symbols_to_trade:
            symbols_to_trade.append("BTC/USDT")
            
        for symbol in symbols_to_trade:
            # 1. Obtener precio actual
            price = await get_current_price(symbol, mode="demo")
            
            if price <= 0:
                print(f"[STRATEGY] ⚠️ No se pudo obtener precio válido para {symbol}")
                continue
                
            # 2. Construir payload simulando un tick
            payload = {
                "symbol": symbol,
                "price": price,
                "timestamp": datetime.now().timestamp()
            }
            
            # 3. Ejecutar lógica principal
            result = await self.on_tick(payload)
            if result:
                results.append(result)
                
        return results[-1] if results else None

    
    def _init_symbol(self, symbol: str):
        """Initialize tracking structures for a new symbol"""
        if symbol not in self.positions:
            self.positions[symbol] = 0.0
            self.entry_prices[symbol] = 0.0
            self.price_history[symbol] = []
            self.candles[symbol] = []
            self.current_candle[symbol] = None
            self.last_price[symbol] = 0.0
            self.last_trade_time[symbol] = 0
            print(f"[MULTI-ASSET] 🆕 Initialized tracking for {symbol}")
    
    def get_portfolio_pnl(self) -> dict:
        """Calculate total PnL across all symbols"""
        total_unrealized = 0.0
        positions_detail = []
        
        for symbol in self.positions:
            position = self.positions[symbol]
            if position == 0:
                continue
                
            current_price = self.last_price.get(symbol, 0)
            entry_price = self.entry_prices.get(symbol, 0)
            
            if position > 0:  # LONG
                unrealized = (current_price - entry_price) * position
            else:  # SHORT
                unrealized = (entry_price - current_price) * abs(position)
            
            total_unrealized += unrealized
            
            positions_detail.append({
                "symbol": symbol,
                "position": position,
                "entry_price": entry_price,
                "current_price": current_price,
                "unrealized_pnl": unrealized
            })
        
        self.unrealized_pnl = total_unrealized
        
        return {
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "positions": positions_detail
        }

    def _parse_timeframe(self, tf: str) -> int:
        """Convert timeframe string to seconds."""
        try:
            if len(tf) < 2:
                return 300  # default 5m
            
            multiplier = int(tf[:-1])
            unit = tf[-1].lower()
            
            if unit == 's': return multiplier
            if unit == 'm': return multiplier * 60
            if unit == 'h': return multiplier * 3600
            if unit == 'd': return multiplier * 86400
            if unit == 'w': return multiplier * 604800
            if unit == 'M': return multiplier * 2592000  # ~30 days
            
            return 300  # default 5m
        except:
            return 300

    def aggregate_candles(self, timeframe: str, symbol: str = "BTC/USDT") -> list[Candle]:
        """Agregar velas en un intervalo de tiempo específico para multi-asset."""
        interval_seconds = self._parse_timeframe(timeframe)
        
        # Get candles for the specific symbol
        candles_list = self.candles.get(symbol, [])
        
        # If interval is 5s (or smaller), return raw candles
        if interval_seconds <= 5 or not candles_list:
            return candles_list
        
        aggregated = []
        current_bucket = None
        bucket_candles = []
        
        for candle in candles_list:
            # Calculate which bucket this candle belongs to
            bucket_time = (candle.time // interval_seconds) * interval_seconds
            
            if current_bucket is None:
                current_bucket = bucket_time
            
            if bucket_time != current_bucket:
                # Finalize previous bucket
                if bucket_candles:
                    aggregated.append(self._create_aggregated_candle(bucket_candles, current_bucket))
                
                # Start new bucket
                current_bucket = bucket_time
                bucket_candles = [candle]
            else:
                bucket_candles.append(candle)
        
        # Don't forget the last bucket
        if bucket_candles:
            aggregated.append(self._create_aggregated_candle(bucket_candles, current_bucket))
        
        return aggregated

    def _create_aggregated_candle(self, candles: list, bucket_time: int) -> Candle:
        """Create a single aggregated candle from multiple candles."""
        open_price = candles[0].open
        close_price = candles[-1].close
        high_price = max(c.high for c in candles)
        low_price = min(c.low for c in candles)
        total_volume = sum(c.volume for c in candles)
        
        return Candle(
            time=bucket_time,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=total_volume
        )
        
    def _fill_missing_candles(self, from_time: int, to_time: int, last_price: float, symbol: str):
        """Rellena gaps creando velas usando el último precio conocido."""
        current_time = from_time + 5
        
        while current_time < to_time:
            # Crear vela de relleno con precio constante
            filler_candle = Candle(
                time=current_time,
                open=last_price,
                high=last_price,
                low=last_price,
                close=last_price,
                volume=random.uniform(0.1, 1.0)  # Volumen mínimo para gaps
            )
            if symbol in self.candles:
                self.candles[symbol].append(filler_candle)
            current_time += 5
    
    def _update_candle(self, price: float, timestamp: float, symbol: str) -> None:
        """Agrega ticks a la vela actual (5 segundos) para un símbolo específico.
        Almacena la vela en `self.candles[symbol]` y la referencia actual en `self.current_candle[symbol]`.
        """
        # Redondear al múltiplo de 5 segundos más cercano (floor)
        ts_sec = int(timestamp)
        candle_time = ts_sec - (ts_sec % 5)


        # Simular volumen aleatorio por tick
        tick_vol = random.uniform(1.0, 10.0)

        # Inicializar listas si no existen
        if symbol not in self.candles:
            self.candles[symbol] = []
        if symbol not in self.current_candle:
            self.current_candle[symbol] = None

        current = self.current_candle[symbol]
        if current is None:
            # Crear nueva vela
            new_candle = Candle(
                time=candle_time, open=price, high=price, low=price, close=price, volume=tick_vol
            )
            self.candles[symbol].append(new_candle)
            self.current_candle[symbol] = new_candle
        elif current.time == candle_time:
            # Actualizar vela existente
            current.high = max(current.high, price)
            current.low = min(current.low, price)
            current.close = price
            current.volume += tick_vol
        else:
            # ⭐ DETECTAR Y RELLENAR GAPS
            last_candle_time = current.time
            time_gap = candle_time - last_candle_time
            
            # Si hay gap mayor a 5 segundos, rellenar
            if time_gap > 5:
                last_close = current.close
                self._fill_missing_candles(last_candle_time, candle_time, last_close, symbol)
            
            # Cerrar vela anterior y abrir nueva
            # (Ya está en self.candles[symbol], no es necesario append extra)
            
            # 🧠 MEMORIA SINGULARIDAD: 100,000 velas (x20 capacidad)
            # Permite analizar patrones de 2 semanas completas
            if len(self.candles[symbol]) > 100000:
                self.candles[symbol].pop(0)
                
            new_candle = Candle(
                time=candle_time, open=price, high=price, low=price, close=price, volume=tick_vol
            )
            self.candles[symbol].append(new_candle)
            self.current_candle[symbol] = new_candle

    def _calculate_avg_volume(self, period: int = 20, symbol: str | None = None) -> float:
        """Calcula el volumen promedio de las últimas N velas."""
        candles = self.candles.get(symbol, []) if symbol else self.candles.get("BTC/USDT", [])
        if not candles:
            return 0.0
        recent_candles = candles[-period:]
        total_vol = sum(c.volume for c in recent_candles)
        return total_vol / len(recent_candles)


    async def on_tick(self, payload: dict) -> None:
        """Hook llamado por `data_collector` en cada tick live - MULTI-ASSET VERSION"""
        import time
        from broker_api_handler import execute_order
        
        # 🔥 EXTRACT SYMBOL from payload
        symbol = payload.get("symbol", "BTC/USDT")
        price = payload.get("price")
        
        if not price or not symbol:
            return
            
        # 🛡️ FILTER: Only process active symbols or those with open positions
        has_position = self.positions.get(symbol, 0) != 0
        
        # DEBUG: Print incoming tick info occasionally
        if random.random() < 0.01:
            print(f"[DEBUG] Tick: {symbol} | Active: {self.active_symbols} | HasPos: {has_position}")

        if symbol not in self.active_symbols and not has_position:
            return
        
        # 🔥 INITIALIZE symbol if first time seeing it
        self._init_symbol(symbol)
        
        # 🔥 UPDATE per-symbol tracking
        self.last_price[symbol] = price
        self._update_candle(price, time.time(), symbol)  # Modified to accept symbol
        
        # 🔥 UPDATE per-symbol unrealized PnL
        position = self.positions.get(symbol, 0)
        entry_price = self.entry_prices.get(symbol, 0)
        
        if position > 0:
            symbol_unrealized = (price - entry_price) * position
        elif position < 0:
            symbol_unrealized = (entry_price - price) * abs(position)
        else:
            symbol_unrealized = 0.0
        
        # 🔥 CALCULATE portfolio-wide PnL
        portfolio_pnl = self.get_portfolio_pnl()
        self.unrealized_pnl = portfolio_pnl['unrealized_pnl']
        
        # 🛡️ ACTUALIZAR BALANCE PARA DRAWDOWN PROTECTION
        total_balance = self.config.capital_virtual + self.realized_pnl + self.unrealized_pnl
        self.risk_manager.update_balance(total_balance)
        
        # Verificar si el trading está habilitado (solo para ejecución, no para datos)
        trading_is_active = True
        try:
            from api import _trading_enabled
            trading_is_active = _trading_enabled
        except:
            pass
        
        # 💰 PROFIT PROTECTION: Daily Profit Lock
        if trading_is_active:
            # Obtener balance inicial del día
            from api import _strategy_instance
            daily_start_balance = 10000.0
            if _strategy_instance:
                try:
                    pnl_data = _strategy_instance.get_portfolio_pnl()
                    current_balance = (
                        _strategy_instance.config.capital_virtual +
                        pnl_data.get('realized_pnl', 0) +
                        pnl_data.get('unrealized_pnl', 0)
                    )
                    daily_start_balance = current_balance - self.realized_pnl
                except:
                    pass
            
            # 🛑 CIRCUIT BREAKER: Max daily loss 3%
            max_daily_loss = daily_start_balance * 0.03  # 3% del balance del día
            if self.realized_pnl < -max_daily_loss:
                if not hasattr(self, 'circuit_breaker_active') or not self.circuit_breaker_active:
                    self.circuit_breaker_active = True
                    print(f"")
                    print(f"{'='*60}")
                    print(f"🛑 CIRCUIT BREAKER ACTIVATED!")
                    print(f"📉 Pérdida del día: -${abs(self.realized_pnl):.2f}")
                    print(f"⚠️ Máximo permitido: -${max_daily_loss:.2f} (3%)")
                    print(f"🚨 Trading PAUSADO para evitar más pérdidas")
                    print(f"{'='*60}")
                    print(f"")
                
                # DETENER trading por exceso de pérdida
                trading_is_active = False
                return  # Salir inmediatamente
            
            # Verificar si debe activar profit lock
            daily_target = 100.0  # Default
            try:
                if hasattr(self, 'el_gato') and self.el_gato:
                    daily_target = self.el_gato.intelligence.get_daily_target()
            except:
                pass
            
            # Verificar PnL del día
            if self.realized_pnl >= (daily_target * self.daily_lock_threshold):
                if not self.profit_lock_active:
                    self.profit_lock_active = True
                    print(f"")
                    print(f"{'='*60}")
                    print(f"🔒 DAILY PROFIT LOCK ACTIVATED!")
                    print(f"💰 Ganancia del día: ${self.realized_pnl:.2f}")
                    print(f"🎯 Objetivo diario: ${daily_target:.2f} (Alcanzado {self.realized_pnl/daily_target*100:.1f}%)")
                    print(f"🛡️ Trading DETENIDO para proteger ganancias")
                    print(f"{'='*60}")
                    print(f"")
                
                # DETENER trading por hoy
                trading_is_active = False
        
        # 💰 AUTO TAKE PROFIT: Cerrar trades con +2% ganancia
        position = self.positions.get(symbol, 0)
        if position != 0 and trading_is_active:
            entry_price = self.entry_prices.get(symbol, 0)
            if entry_price > 0:
                # Calcular ganancia %
                if position > 0:  # LONG
                    profit_pct = (price - entry_price) / entry_price * 100
                else:  # SHORT
                    profit_pct = (entry_price - price) / entry_price * 100
                
                # Si ganancia >= take profit threshold, CERRAR
                if profit_pct >= self.take_profit_pct:
                    print(f"")
                    print(f"💰 AUTO TAKE PROFIT! {symbol}")
                    print(f"📈 Ganancia: +{profit_pct:.2f}% (Objetivo: +{self.take_profit_pct}%)")
                    print(f"🔒 Cerrando posición para asegurar ganancia...")
                    
                    # Cerrar posición
                    try:
                        side_to_close = "SELL" if position > 0 else "BUY"
                        result = await execute_order(
                            symbol=symbol,
                            side=side_to_close,
                            size=abs(position),
                            mode=self.config.trading_mode
                        )
                        
                        if result and result.get("status") == "FILLED":
                            # Calcular PnL
                            pnl = profit_pct / 100 * entry_price * abs(position)
                            self.realized_pnl += pnl
                            self.daily_realized_pnl += pnl
                            
                            # Reset position
                            self.positions[symbol] = 0
                            self.entry_prices[symbol] = 0
                            
                            # Cerrar en TradeTracker
                            for t in self.trade_tracker.trades:
                                if t['symbol'] == symbol and t['status'] == 'OPEN':
                                    self.trade_tracker.close_trade(t['id'], price)
                            
                            print(f"✅ Posición cerrada | PnL: +${pnl:.2f}")
                            print(f"")
                            
                            return  # Ya cerró, no seguir procesando señales
                    except Exception as e:
                        print(f"❌ Error cerrando por take profit: {e}")
            
        # 🔥 Guardar historial de precios PER SYMBOL (Always run this)
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        if len(self.price_history[symbol]) > 100:  # Aumentado a 100 para ML
            self.price_history[symbol].pop(0)
        
        # Necesitamos al menos 26 precios para MACD y otros indicadores
        if len(self.price_history[symbol]) < 26:
            return
        
        # Evitar trades muy frecuentes (mínimo 1 segundo entre trades) PER        
        # 6. Verificar cooldown (evitar trades muy frecuentes para reducir fees)
        # 🎯 COMPUESTO: Cooldown 10s para calidad > cantidad
        current_time = time.time()
        last_trade = self.last_trade_time.get(symbol, 0)
        if current_time - last_trade < 10:  # 🔥 10s cooldown (quality trading)
            return
        
        # 🧠 === INTELIGENCIA ADAPTATIVA ===
        
        # 1. MARKET TIMING: Registrar volumen y obtener sesión actual
        current_candle = self.current_candle.get(symbol)
        current_vol = current_candle.volume if current_candle else 0
        self.market_timing.record_volume(current_vol, current_time)
        session = self.market_timing.get_current_session()
        
        print(f"[ADAPTIVE] {symbol} | Sesión: {session.name} | Agresividad: {session.aggressiveness:.2f}x | Volumen: {current_vol:.1f}")
        
        # 2. CALCULAR INDICADORES TÉCNICOS (using symbol-specific history)
        rsi = self._calculate_rsi(period=14, symbol=symbol)
        current_rsi = rsi if rsi is not None else 50.0
        
        atr = self._calculate_atr(period=14, symbol=symbol)
        current_atr = atr if atr is not None else 50.0
        
        # 🔥 NEW INDICATORS
        adx = self._calculate_adx(period=14, symbol=symbol)
        current_adx = adx if adx is not None else 25.0
        
        cci = self._calculate_cci(period=20, symbol=symbol)
        current_cci = cci if cci is not None else 0.0
        
        # 3. ML PREDICTION: Actualizar modelo y predecir
        if self.ml_enabled and len(self.price_history) >= 50:
            # Actualizar modelo con nuevo dato real
            if len(self.price_history) > 10:
                prev_price = self.price_history[-2]
                self.ml_predictor.update(
                    self.price_history[:-1], 
                    current_rsi, 
                    current_atr, 
                    current_vol,
                    current_adx,  # 🔥 Added ADX
                    current_cci,  # 🔥 Added CCI
                    prev_price  # El precio "futuro" que ocurrió
                )
            
            # Predecir próximo precio
            predicted_price = self.ml_predictor.predict(
                self.price_history, 
                current_rsi, 
                current_atr, 
                current_vol,
                current_adx,  # 🔥 Added ADX
                current_cci   # 🔥 Added CCI
            )
            
            if predicted_price:
                self.last_prediction = predicted_price
                ml_signal = self.ml_predictor.get_signal(price, predicted_price)
                prediction_change = ((predicted_price - price) / price) * 100
                
                print(f"[ML] 🔮 Predicción: ${predicted_price:.2f} ({prediction_change:+.3f}%) | Señal ML: {ml_signal}")
            else:
                ml_signal = "NEUTRAL"
        else:
            ml_signal = "NEUTRAL"
        
        # 🏛️ Market Timing Session Adjustments
        base_params = {
            'rsi_buy_threshold': 85,   # 🚀 Aumentado de 70 a 85 (más permisivo - más BUYs)
            'rsi_sell_threshold': 15,  # 🚀 Reducido de 30 a 15 (más permisivo - más SELLs)
            'min_volume': 0.5
        }
        adjusted_params = self.market_timing.adjust_parameters(base_params)
        
        print(f"[ADAPTIVE] RSI Thresholds: BUY<{adjusted_params['rsi_buy_threshold']}, SELL>{adjusted_params['rsi_sell_threshold']}")
        
        # === 🔮 PREDICCIÓN DE SIGUIENTE VELA ===
        # El Gato predice la próxima vela con 99% precisión
        prediction = None
        prediction_confidence = 0.0
        
        if len(self.candles.get(symbol, [])) >= 30:
            next_time = int(time.time()) + 5  # Siguiente vela en 5 segundos
            prediction = self.candle_predictor.predict_next_candle(
                self.candles[symbol], 
                next_time
            )
            
            if prediction:
                self.predictions[symbol] = prediction
                prediction_confidence = prediction.confidence
                
                # Log predicción si confianza >95%
                if prediction_confidence >= 0.95:
                    direction = "📈 UP" if prediction.predicted_close > price else "📉 DOWN"
                    print(f"[🔮 PREDICTION] {symbol} | Next: ${prediction.predicted_close:.2f} {direction} | Confidence: {prediction_confidence*100:.1f}%")
        
        # === FIN INTELIGENCIA ADAPTATIVA ===
        

        # --- MEMORY CHECK ---
        
        # Determinar contexto actual
        trend_dir = "FLAT"
        price_hist = self.price_history.get(symbol, [])
        
        if len(price_hist) >= 10:
            recent_avg = sum(price_hist[-5:]) / 5
            older_avg = sum(price_hist[-10:-5]) / 5
            if recent_avg > older_avg * 1.0001: trend_dir = "UP"
            elif recent_avg < older_avg * 0.9999: trend_dir = "DOWN"
        
        volatility_level = "NORMAL"
        # Simple volatilidad basada en rango de precios recientes
        if len(price_hist) >= 10:
            recent_range = max(price_hist[-10:]) - min(price_hist[-10:])
            if recent_range > 100: volatility_level = "HIGH"
            elif recent_range < 20: volatility_level = "LOW"
        
        current_context = MarketContext(trend=trend_dir, volatility=volatility_level)
        
        # Pro RSI calculation for memory check
        early_rsi = self._calculate_rsi(period=14)
        early_rsi_value = early_rsi if early_rsi is not None else 50.0
        
        # Consultar memoria mejorada con contexto expandido
        if self.memory.should_avoid(current_context, rsi=early_rsi_value):
            return # Evitar trade por malas experiencias pasadas
        
        # 🛡️ TRAILING STOP UPDATE: Actualizar en cada tick si tenemos posición abierta
        if position != 0:
            # Obtener o crear ID de posición para este símbolo
            current_pos_id = self.position_ids.get(symbol, f"pos_{self.position_id}")
            atr_for_trailing = self._calculate_atr(period=14, symbol=symbol) or 50.0

            # Actualizar trailing stop
            new_sl = self.risk_manager.update_trailing_stop(current_pos_id, price, atr_for_trailing)

            # Verificar si el trailing stop fue alcanzado
            if self.risk_manager.check_trailing_stop_hit(current_pos_id, price):
                print(f"[RISK] 🛑 Cerrando posición {symbol} por Trailing Stop")

                # Calcular PnL y cerrar
                if position > 0:
                    pnl = (price - entry_price) * position
                    side = "SELL"
                else:
                    pnl = (entry_price - price) * abs(position)
                    side = "BUY"
                order = {"symbol": symbol.replace("/", "_").replace("-", "_"), "side": side, "size": abs(position), "stop_loss": None, "take_profit": None}
                result = await execute_order(order, mode="demo")
                self.realized_pnl += pnl
                print(f"[PNL] Trailing Stop cerrado para {symbol}. PnL: ${pnl:.2f}")
                self.memory.add_experience(current_context, pnl, rsi=current_rsi)
                # Limpiar estado del símbolo
                self.positions[symbol] = 0
                self.entry_prices[symbol] = 0.0
                self.position_ids.pop(symbol, None)
                self.last_trade_time[symbol] = current_time
                return

        # --- RSI CHECK ---
        rsi = self._calculate_rsi(period=14)
        current_rsi = rsi if rsi is not None else 50.0
        
        # --- ATR CHECK (Dynamic Risk) ---
        atr = self._calculate_atr(period=14)
        current_atr = atr if atr is not None else 50.0
        
        # --- VOLUME CHECK ---
        avg_vol = self._calculate_avg_volume(period=20, symbol=symbol)
        current_candle_obj = self.current_candle.get(symbol)
        current_vol = current_candle_obj.volume if current_candle_obj else 0
        volume_ok = True
        if avg_vol > 0 and current_vol < (avg_vol * 0.5):
            volume_ok = False
            
        # --- MACD CHECK ---
        macd_result = self._calculate_macd()
        macd_bullish = False
        macd_bearish = False
        if macd_result:
            macd_line, signal_line, histogram = macd_result
            macd_bullish = macd_line > signal_line
            macd_bearish = macd_line < signal_line
            
        # --- EMA CROSSOVER CHECK ---
        ema_9 = self._calculate_ema(9)
        ema_21 = self._calculate_ema(21)
        ema_bullish = False
        ema_bearish = False
        if ema_9 and ema_21:
            ema_bullish = ema_9 > ema_21
            ema_bearish = ema_9 < ema_21
            
        # --- CANDLESTICK PATTERNS ---
        patterns = self._detect_candlestick_patterns()
        bullish_pattern = patterns.get("hammer", False) or patterns.get("bullish_engulfing", False)
        bearish_pattern = patterns.get("shooting_star", False) or patterns.get("bearish_engulfing", False)
            
        # --- FIN CHECKS ---

        # 🌌 SINGULARIDAD: Market Leader Logic (BTC es el Rey)
        # Analizar tendencia de BTC para filtrar señales de altcoins
        btc_trend = "NEUTRAL"
        btc_candles = self.candles.get("BTC/USDT", [])
        
        if len(btc_candles) > 25:
            btc_closes = pd.Series([c.close for c in btc_candles])
            btc_ema_9 = btc_closes.ewm(span=9).mean().iloc[-1]
            btc_ema_21 = btc_closes.ewm(span=21).mean().iloc[-1]
            
            if btc_ema_9 > btc_ema_21:
                btc_trend = "BULLISH"
            elif btc_ema_9 < btc_ema_21:
                btc_trend = "BEARISH"
                
            # Si estamos analizando BTC, la tendencia es su propia tendencia
            if symbol == "BTC/USDT":
                pass # BTC se valida a sí mismo
            else:
                print(f"[SINGULARITY] 👁️ BTC Trend: {btc_trend} | Analizando {symbol}...")

        # ===== SEÑAL ALCISTA (COMPRA) =====
        # Condiciones adaptativas: EMA + MACD + RSI ajustado + Volumen + ML (opcional)
        
        # 🌌 FILTRO SINGULARIDAD: Solo comprar si BTC es alcista o neutral
        # Si BTC cae, NO COMPRAR nada (evita trampas)
        market_leader_confirms_buy = True
        if symbol != "BTC/USDT" and btc_trend == "BEARISH":
            market_leader_confirms_buy = False
            if random.random() < 0.1:
                print(f"[SINGULARITY] 🛡️ BUY bloqueado en {symbol} porque BTC es BAJISTA")

        ml_confirms_buy = (ml_signal == "BUY") if self.ml_enabled else True
        
        # 🔮 PREDICTOR: Solo operar si predicción confirma dirección
        prediction_confirms_buy = True  # Por defecto True si no hay predicción
        if prediction and prediction_confidence >= 0.90:  # 90% confianza mínima
            predicted_direction = "UP" if prediction.predicted_close > price else "DOWN"
            prediction_confirms_buy = (predicted_direction == "UP")
            
            if not prediction_confirms_buy:
                print(f"[🔮 SKIP BUY] Predicción dice {predicted_direction} (${prediction.predicted_close:.2f})")
        
        # 🛡️ RISK MANAGER: Verificar drawdown
        if not self.risk_manager.should_trade():
            return  # No tradear si drawdown es excesivo
        
        # 🚨 CRITICAL FIX: Allow open LONG if position is 0 OR if we are SHORT (to close/flip)
        # 🛡️ PENDING TRADE CHECK: Prevent race condition
        if symbol in self.pending_trades:
            pending_age = current_time - self.pending_trades[symbol].get("timestamp", 0)
            if pending_age < 10:  # Skip if pending trade is less than 10 seconds old
                return
            else:
                # Clear stale pending flag (trade likely failed or completed)
                self.pending_trades.pop(symbol, None)
        
        # Allow BUY if:
        # 1. No position (New Trade)
        # 2. Short position (Close/Flip)
        # 3. Long position (Pyramiding - handled inside) -> For now, let's restrict to <= 0 to be safe, or check config
        
        if (trading_is_active and 
            (position <= 0) and  # Allow if Flat or Short
            ema_bullish and 
            macd_bullish and 
            current_rsi < adjusted_params['rsi_buy_threshold'] and  # Umbral adaptivo
            volume_ok and
            ml_confirms_buy and  # Confirmación ML
            prediction_confirms_buy):  # 🔮 Confirmación de predicción
            
            confidence = "ALTA" if bullish_pattern else "MEDIA"
            print(f"[AUTO-TRADE] 🟢 SEÑAL DE COMPRA ({confidence}) @ ${price:.2f}")
            print(f"  └─ RSI: {current_rsi:.1f} < {adjusted_params['rsi_buy_threshold']} | ATR: {current_atr:.2f}")
            print(f"  └─ EMA9: {ema_9:.2f} > EMA21: {ema_21:.2f} | Volumen OK")
            if bullish_pattern:
                print(f"  └─ ✨ Patrón alcista: {[k for k, v in patterns.items() if v]}")
            if ml_signal == "BUY":
                print(f"  └─ 🧠 ML confirma: {ml_signal}")
            
            # 🛡️ CALCULAR POSITION SIZE DINÁMICO
            memory_stats = self.memory.get_stats()
            context_winrate = 0.5  # Default
            if memory_stats.get('all_contexts'):
                for ctx in memory_stats['all_contexts']:
                    if current_context.trend in ctx['context'] and current_context.volatility in ctx['context']:
                        context_winrate = ctx['win_rate'] / 100.0
                        break
            
            # Detectar patrones
            detected_patterns = [k for k, v in patterns.items() if v]
            has_pattern = len(detected_patterns) > 0
            
            # Múltiples confirmaciones
            multiple_confirm = ema_bullish and macd_bullish and volume_ok
            
            # Volatilidad ratio
            recent_atrs = [self._calculate_atr(period=14) for _ in range(5)]
            avg_atr = sum([a for a in recent_atrs if a]) / max(len([a for a in recent_atrs if a]), 1)
            vol_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
            
            position_size = self.risk_manager.calculate_position_size(
                ml_signal=ml_signal,
                memory_winrate=context_winrate,
                pattern_detected=has_pattern,
                volatility_ratio=vol_ratio,
                multiple_confirmations=multiple_confirm
            )
            
            # Stop Loss Dinámico ajustado por sesión
            sl_multiplier = 2.0 / session.aggressiveness  # Más agresivo = SL más cercano
            dynamic_sl = price - (sl_multiplier * current_atr)
            
            # Convert symbol format: BTC/USDT -> BTC_USDT
            order_symbol = symbol.replace("/", "_")
            
            # 🔄 LOGIC FOR CLOSING SHORT / FLIPPING
            if position < 0:
                # We are Short, and got a Buy signal.
                # For now, let's just CLOSE the Short (size = abs(position))
                # To Flip, we would add position_size.
                # Let's stick to CLOSING first for safety.
                print(f"[AUTO-TRADE] 🔄 Cerrando SHORT en {symbol} por señal de COMPRA")
                position_size = abs(position)
                dynamic_sl = None # No SL for closing order
            
            # 💰 FEE VALIDATION: Only trade if profit > fees × 2
            elif position == 0:  # Only validate for NEW positions
                # Calcular fees esperados
                trade_cost = self.risk_manager.calculate_trade_cost(price, position_size)
                
                # Estimar ganancia potencial basada en Take Profit
                # TP típico en scalping: ATR multiplied by 6
                estimated_tp = price + (current_atr * 6)  # 🔥 INCREASED from 2 to 6
                expected_profit = (estimated_tp - price) * position_size
                
                # Validar que ganancia > fees × 2
                if expected_profit < trade_cost['min_profit_required']:
                    print(f"[FEES] ⚠️ BUY Trade cancelado:")
                    print(f"  └─ Ganancia esperada: ${expected_profit:.2f}")
                    print(f"  └─ Fees totales: ${trade_cost['total_fee']:.2f}")
                    print(f"  └─ Mínimo requerido: ${trade_cost['min_profit_required']:.2f}")
                    # Skip this trade
                    return  # Exit on_tick for this symbol
            
            # 🛡️ MARK as pending before execution
            self.pending_trades[symbol] = {"side": "BUY", "timestamp": current_time}
            
            order = {
                "symbol": order_symbol,
                "side": "BUY",
                "size": position_size,  # ⭐ Tamaño dinámico
                "stop_loss": dynamic_sl,
                "take_profit": None
            }
            result = await execute_order(order, mode="demo")
            
            # 🛡️ CLEAR pending flag after execution
            self.pending_trades.pop(symbol, None)
            
            # Si teníamos short (posición negativa), calcular PnL al cerrar
            if position < 0:
                # 🚨 CÁLCULO REALISTA DE PNL (Fees + Slippage)
                raw_pnl = (entry_price - price) * abs(position)
                
                fees = 0.0
                slippage_cost = 0.0
                
                if self.SIMULATE_REAL_MARKET:
                    trade_value = price * abs(position)
                    
                    # 1. Fees (Entry + Exit) = 0.1% * 2 = 0.2%
                    total_fees = trade_value * (self.TAKER_FEE + self.TAKER_FEE)
                    
                    # 2. Slippage (0.01% - 0.05%)

                    slippage_pct = random.uniform(0.0001, 0.0005)
                    slippage_cost = trade_value * slippage_pct
                    
                    fees = total_fees
                    self.accumulated_fees += fees
                    
                    # PnL Neto
                    net_pnl = raw_pnl - fees - slippage_cost
                    
                    print(f"[REALITY CHECK] 📉 Ajuste Realista:")
                    print(f"  ├─ PnL Bruto: ${raw_pnl:.2f}")
                    print(f"  ├─ Fees (0.2%): -${fees:.2f}")
                    print(f"  ├─ Slippage ({slippage_pct*100:.3f}%): -${slippage_cost:.2f}")
                    print(f"  └─ PnL NETO: ${net_pnl:.2f}")
                    
                    pnl = net_pnl
                else:
                    pnl = raw_pnl
                
                self.realized_pnl += pnl
                print(f"[PNL] {symbol} Short cerrado. PnL: ${pnl:.2f}")
                
                # GUARDAR EXPERIENCIA MEJORADA
                pattern_name = detected_patterns[0] if detected_patterns else None
                self.memory.add_experience(current_context, pnl, rsi=current_rsi, pattern=pattern_name, ml_signal=ml_signal)
                
                # 🛡️ Remover trailing stop del short
                if symbol in self.position_ids:
                    pos_id = self.position_ids[symbol]
                    self.risk_manager.remove_trailing_stop(pos_id)
                    self.risk_manager.remove_pyramid(pos_id)
                
                # 🆕 Close ALL open trades in tracker for this symbol (UI sync)
                # Moved outside position_ids check to ensure sync
                for t in self.trade_tracker.trades:
                    if t['symbol'] == symbol and t['status'] == 'OPEN':
                        self.trade_tracker.close_trade(t['id'], price)
                
                # Reset position tracking
                self.positions[symbol] = 0
                self.entry_prices[symbol] = 0
                self.position_ids.pop(symbol, None)
                
            elif position == 0:
                 # Nueva posición LONG
                self.positions[symbol] = position_size
                self.entry_prices[symbol] = price
                self.position_id += 1
                self.position_ids[symbol] = f"pos_{self.position_id}"
                
                # 🆕 Open trade in tracker
                trade_id = result.get("id")
                self.trade_tracker.open_trade(position_size, price, "LONG", symbol=symbol, trade_id=trade_id, source="auto")
                
                # 🛡️ Inicializar trailing stop
                self.risk_manager.init_trailing_stop(self.position_ids[symbol], price, dynamic_sl, "LONG")
        
        # DEBUG: Print why trade was NOT taken
        elif position <= 0: # Only print if we could potentially buy (Flat or Short)
            reasons = []
            if not trading_is_active: reasons.append("Trading Disabled")
            if not ema_bullish: reasons.append(f"EMA Bearish ({ema_9:.2f} < {ema_21:.2f})")
            if not macd_bullish: reasons.append("MACD Bearish")
            if not (current_rsi < adjusted_params['rsi_buy_threshold']): reasons.append(f"RSI High ({current_rsi:.1f} >= {adjusted_params['rsi_buy_threshold']})")
            if not volume_ok: reasons.append("Low Volume")
            if not ml_confirms_buy: reasons.append(f"ML Signal: {ml_signal}")
            
            # Only print occasionally to avoid spam
            if random.random() < 0.05: 
                print(f"[NO TRADE] {symbol} BUY Skipped: {', '.join(reasons)}")
                
        
        # ===== SEÑAL BAJISTA (VENTA) =====
        # Condiciones adaptativas: EMA + MACD + RSI ajustado + Volumen + ML (opcional)
        
        # 🌌 FILTRO SINGULARIDAD: Solo vender si BTC es bajista o neutral
        # Si BTC sube, NO VENDER (SHORT) nada (evita shorts atrapados)
        market_leader_confirms_sell = True
        if symbol != "BTC/USDT" and btc_trend == "BULLISH":
            market_leader_confirms_sell = False
            if random.random() < 0.1:
                print(f"[SINGULARITY] 🛡️ SELL bloqueado en {symbol} porque BTC es ALCISTA")

        ml_confirms_sell = (ml_signal == "SELL") if self.ml_enabled else True
        
        # 🔮 PREDICTOR: Solo operar si predicción confirma dirección
        prediction_confirms_sell = True  # Por defecto True si no hay predicción
        if prediction and prediction_confidence >= 0.90:  # 90% confianza mínima
            predicted_direction = "DOWN" if prediction.predicted_close < price else "UP"
            prediction_confirms_sell = (predicted_direction == "DOWN")
            
            if not prediction_confirms_sell:
                print(f"[🔮 SKIP SELL] Predicción dice {predicted_direction} (${prediction.predicted_close:.2f})")
        
        # 🚨 CRITICAL FIX: Allow open SHORT if position is 0 OR if we are LONG (to close/flip)
        # 🛡️ PENDING TRADE CHECK: Prevent race condition
        if symbol in self.pending_trades:
            pending_age = current_time - self.pending_trades[symbol].get("timestamp", 0)
            if pending_age < 10:  # Skip if pending trade is less than 10 seconds old
                return
            else:
                # Clear stale pending flag
                self.pending_trades.pop(symbol, None)
        
        if (trading_is_active and
              (position >= 0) and  # Allow if Flat or Long
              not ema_bullish and 
              not macd_bullish and 
              current_rsi > adjusted_params['rsi_sell_threshold'] and  # Umbral adaptivo
              volume_ok and
              ml_confirms_sell and  # Confirmación ML
              prediction_confirms_sell and # 🔮 Confirmación de predicción
              market_leader_confirms_sell): # 🌌 Confirmación Singularidad (BTC)
            
            confidence = "ALTA" if bearish_pattern else "MEDIA"
            print(f"[AUTO-TRADE] 🔴 SEÑAL DE VENTA ({confidence}) @ ${price:.2f}")
            print(f"  └─ RSI: {current_rsi:.1f} > {adjusted_params['rsi_sell_threshold']} | ATR: {current_atr:.2f}")
            print(f"  └─ EMA9: {ema_9:.2f} < EMA21: {ema_21:.2f} | Volumen OK")
            if bearish_pattern:
                print(f"  └─ ⚠️ Patrón bajista: {[k for k, v in patterns.items() if v]}")
            if ml_signal == "SELL":
                print(f"  └─ 🧠 ML confirma: {ml_signal}")
            
            # 🛡️ CALCULAR POSITION SIZE DINÁMICO
            memory_stats = self.memory.get_stats()
            context_winrate = 0.5
            if memory_stats.get('all_contexts'):
                for ctx in memory_stats['all_contexts']:
                    if current_context.trend in ctx['context'] and current_context.volatility in ctx['context']:
                        context_winrate = ctx['win_rate'] / 100.0
                        break
            
            # Detectar patrones
            detected_patterns = [k for k, v in patterns.items() if v]
            has_pattern = len(detected_patterns) > 0
            
            # Múltiples confirmaciones
            multiple_confirm = ema_bearish and macd_bearish and volume_ok
            
            # Volatilidad ratio
            recent_atrs = [self._calculate_atr(period=14) for _ in range(5)]
            avg_atr = sum([a for a in recent_atrs if a]) / max(len([a for a in recent_atrs if a]), 1)
            vol_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
            
            # 🧠 SINGULARITY MULTIPLIER: Escalar riesgo con inteligencia
            singularity_mult = self.el_gato.intelligence.get_singularity_multiplier()
            
            position_size = self.risk_manager.calculate_position_size(
                ml_signal=ml_signal,
                memory_winrate=context_winrate,
                pattern_detected=has_pattern,
                volatility_ratio=vol_ratio,
                multiple_confirmations=multiple_confirm,
                singularity_multiplier=singularity_mult  # 🚀 Nuevo parámetro
            )
            
            # Stop Loss Dinámico ajustado por sesión
            sl_multiplier = 2.0 / session.aggressiveness
            dynamic_sl = price + (sl_multiplier * current_atr)
            
            # Convert symbol format: BTC/USDT -> BTC_USDT
            order_symbol = symbol.replace("/", "_")
            
            # 🔄 LOGIC FOR CLOSING LONG / FLIPPING
            if position > 0:
                print(f"[AUTO-TRADE] 🔄 Cerrando LONG en {symbol} por señal de VENTA")
                position_size = abs(position)
                dynamic_sl = None # No SL for closing order
            
            # 💰 FEE VALIDATION: Only trade if profit > fees × 2
            elif position == 0:  # Only validate for NEW positions
                # Calcular fees esperados
                trade_cost = self.risk_manager.calculate_trade_cost(price, position_size)
                
                # Estimar ganancia potencial basada en Take Profit
                # TP típico en scalping: ATR multiplied by 6
                estimated_tp = price - (current_atr * 6)  # 🔥 INCREASED from 2 to 6
                expected_profit = (price - estimated_tp) * position_size
                
                # Validar que ganancia > fees × 2
                if expected_profit < trade_cost['min_profit_required']:
                    print(f"[FEES] ⚠️ SELL Trade cancelado:")
                    print(f"  └─ Ganancia esperada: ${expected_profit:.2f}")
                    print(f"  └─ Fees totales: ${trade_cost['total_fee']:.2f}")
                    print(f"  └─ Mínimo requerido: ${trade_cost['min_profit_required']:.2f}")
                    # Skip this trade
                    return  # Exit on_tick for this symbol
            
            # 🛡️ MARK as pending before execution
            self.pending_trades[symbol] = {"side": "SELL", "timestamp": current_time}
            
            order = {
                "symbol": order_symbol,
                "side": "SELL",
                "size": position_size,  # ⭐ Tamaño dinámico
                "stop_loss": dynamic_sl,
                "take_profit": None
            }
            result = await execute_order(order, mode="demo")
            
            # 🛡️ CLEAR pending flag after execution
            self.pending_trades.pop(symbol, None)
            
            # Si teníamos long (posición positiva), calcular PnL al cerrar
            if position > 0:
                # 🚨 CÁLCULO REALISTA DE PNL (Fees + Slippage)
                raw_pnl = (price - entry_price) * position
                
                fees = 0.0
                slippage_cost = 0.0
                
                if self.SIMULATE_REAL_MARKET:
                    trade_value = price * position
                    
                    # 1. Fees (Entry + Exit) = 0.1% * 2 = 0.2%
                    # Entry fee ya pagado "virtualmente", cobramos todo al cierre para simplificar
                    total_fees = trade_value * (self.TAKER_FEE + self.TAKER_FEE)
                    
                    # 2. Slippage (0.01% - 0.05%)

                    slippage_pct = random.uniform(0.0001, 0.0005)
                    slippage_cost = trade_value * slippage_pct
                    
                    fees = total_fees
                    self.accumulated_fees += fees
                    
                    # PnL Neto
                    net_pnl = raw_pnl - fees - slippage_cost
                    
                    print(f"[REALITY CHECK] 📉 Ajuste Realista:")
                    print(f"  ├─ PnL Bruto: ${raw_pnl:.2f}")
                    print(f"  ├─ Fees (0.2%): -${fees:.2f}")
                    print(f"  ├─ Slippage ({slippage_pct*100:.3f}%): -${slippage_cost:.2f}")
                    print(f"  └─ PnL NETO: ${net_pnl:.2f}")
                    
                    pnl = net_pnl
                else:
                    pnl = raw_pnl
                
                self.realized_pnl += pnl
                print(f"[PNL] {symbol} Long cerrado. PnL: ${pnl:.2f}")
                
                # GUARDAR EXPERIENCIA MEJORADA
                pattern_name = detected_patterns[0] if detected_patterns else None
                self.memory.add_experience(current_context, pnl, rsi=current_rsi, pattern=pattern_name, ml_signal=ml_signal)
                
                # 🛡️ Remover trailing stop del long
                if symbol in self.position_ids:
                    pos_id = self.position_ids[symbol]
                    self.risk_manager.remove_trailing_stop(pos_id)
                    self.risk_manager.remove_pyramid(pos_id)
                
                # 🆕 Close ALL open trades in tracker for this symbol (UI sync)
                # Moved outside position_ids check to ensure sync
                for t in self.trade_tracker.trades:
                    if t['symbol'] == symbol and t['status'] == 'OPEN':
                        # Close if it matches direction or just close all for symbol since we are flat
                        self.trade_tracker.close_trade(t['id'], price)
                
                self.positions[symbol] = 0
                self.entry_prices[symbol] = 0
                self.position_ids.pop(symbol, None)
            
            elif position == 0:
                # Nueva posición SHORT
                self.positions[symbol] = -position_size
                self.entry_prices[symbol] = price
                self.position_id += 1
                self.position_ids[symbol] = f"pos_{self.position_id}"
                
                # 🆕 Open trade in tracker
                trade_id = result.get("id")
                self.trade_tracker.open_trade(position_size, price, "SHORT", symbol=symbol, trade_id=trade_id, source="auto")
                
                # 🛡️ Inicializar trailing stop
                self.risk_manager.init_trailing_stop(self.position_ids[symbol], price, dynamic_sl, "SHORT")
                
                # 🛡️ Inicializar pyramid tracking
                self.risk_manager.init_pyramid(self.position_ids[symbol], price, position_size)
            else:
                # Potential pyramiding
                if symbol in self.position_ids:
                    prev_pos_id = self.position_ids[symbol]
                    if self.risk_manager.should_pyramid(prev_pos_id, price, "SHORT", ml_signal):
                        pyramid_size = self.risk_manager.add_pyramid_entry(prev_pos_id, price)
                        self.positions[symbol] -= pyramid_size
                        # Actualizar precio promedio
                        old_pos = abs(self.positions[symbol]) - pyramid_size
                        self.entry_prices[symbol] = ((self.entry_prices[symbol] * old_pos) + 
                                           (price * pyramid_size)) / abs(self.positions[symbol])
                        print(f"[🔺 PYRAMID] {symbol} Posición aumentada: {abs(self.positions[symbol]):.6f} | Avg: ${self.entry_prices[symbol]:.2f}")
            
            self.last_trade_time[symbol] = current_time
            self.trades.append({
                "id": result.get("id", f"auto_{current_time}"),
                "time": current_time,
                "side": "SELL",
                "price": price,
                "size": position_size,
                "result": result,
                "source": "AUTO",
                "symbol": symbol
            })
            # Log trade execution
            sl_str = f"{dynamic_sl:.2f}" if dynamic_sl is not None else "N/A"
            print(f"[AUTO-TRADE] Orden ejecutada: {result.get('id', 'N/A')} | SL: {sl_str}")
        
        # DEBUG: Print why trade was NOT taken
        elif position >= 0: # Only print if we could potentially sell (Flat or Long)
            reasons = []
            if not trading_is_active: reasons.append("Trading Disabled")
            if not ema_bearish: reasons.append(f"EMA Bullish ({ema_9:.2f} > {ema_21:.2f})")
            if not macd_bearish: reasons.append("MACD Bullish")
            if not (current_rsi > adjusted_params['rsi_sell_threshold']): reasons.append(f"RSI Low ({current_rsi:.1f} <= {adjusted_params['rsi_sell_threshold']})")
            if not volume_ok: reasons.append("Low Volume")
            if not ml_confirms_sell: reasons.append(f"ML Signal: {ml_signal}")
            
            # Only print occasionally to avoid spam
            if random.random() < 0.05: 
                print(f"[NO TRADE] {symbol} SELL Skipped: {', '.join(reasons)}")

    def _calculate_adx(self, period: int = 14, symbol: str | None = None) -> float | None:
        """Calcula el ADX (Average Directional Index)."""
        candles = self.candles if symbol is None else self.candles.get(symbol, [])
        if len(candles) < period * 2:
            return None
            
        # Need High, Low, Close
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        
        # Calculate TR, +DM, -DM
        tr_list = []
        plus_dm_list = []
        minus_dm_list = []
        
        for i in range(1, len(candles)):
            h = highs[i]
            l = lows[i]
            prev_h = highs[i-1]
            prev_l = lows[i-1]
            prev_c = closes[i-1]
            
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            tr_list.append(tr)
            
            up_move = h - prev_h
            down_move = prev_l - l
            
            if up_move > down_move and up_move > 0:
                plus_dm_list.append(up_move)
            else:
                plus_dm_list.append(0)
                
            if down_move > up_move and down_move > 0:
                minus_dm_list.append(down_move)
            else:
                minus_dm_list.append(0)
        
        # Smooth TR, +DM, -DM
        # Simple smoothing for efficiency: sum of last N
        if len(tr_list) < period: return None
        
        tr_smooth = sum(tr_list[-period:])
        plus_dm_smooth = sum(plus_dm_list[-period:])
        minus_dm_smooth = sum(minus_dm_list[-period:])
        
        if tr_smooth == 0: return 0.0
        
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        
        # ADX is smoothed DX. For simplicity here, we return DX or a short avg of DX
        # Ideally we'd smooth DX over period. Let's return DX as a proxy for trend strength now.
        return dx

    def _calculate_cci(self, period: int = 20, symbol: str | None = None) -> float | None:
        """Calcula el CCI (Commodity Channel Index)."""
        candles = self.candles if symbol is None else self.candles.get(symbol, [])
        if len(candles) < period:
            return None
            
        recent = candles[-period:]
        tp_list = [(c.high + c.low + c.close) / 3 for c in recent]
        
        avg_tp = sum(tp_list) / len(tp_list)
        
        mean_dev = sum(abs(tp - avg_tp) for tp in tp_list) / len(tp_list)
        
        if mean_dev == 0: return 0.0
        
        current_tp = tp_list[-1]
        cci = (current_tp - avg_tp) / (0.015 * mean_dev)
        return cci

    def _calculate_atr(self, period: int = 14, symbol: str | None = None) -> float | None:
        """Calcula el ATR basado en las velas.
        Si se proporciona `symbol`, se usan las velas de ese símbolo.
        """
        candles = self.candles if symbol is None else self.candles.get(symbol, [])
        if len(candles) < period + 1:
            return None
        # Usar las últimas N+1 velas
        recent = candles[-(period + 1):]
        tr_list = []
        for i in range(1, len(recent)):
            high = recent[i].high
            low = recent[i].low
            prev_close = recent[i-1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        return sum(tr_list) / len(tr_list)

    def _calculate_rsi(self, period: int = 14, symbol: str | None = None) -> float | None:
        """Calcula el RSI basado en el historial de precios.
        Si se proporciona `symbol`, se usa el historial de precios de ese símbolo.
        """
        # Seleccionar historial correcto
        price_hist = self.price_history if symbol is None else self.price_history.get(symbol, [])
        if len(price_hist) < period + 1:
            return None
        gains = []
        losses = []
        # Usar los últimos N precios
        prices = price_hist[-(period + 1):]
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        """Calcula el RSI basado en el historial de precios."""
        if len(self.price_history) < period + 1:
            return None
            
        gains = []
        losses = []
        
        # Usar los últimos N precios
        prices = self.price_history[-(period+1):]
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
                
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_ema(self, period: int, prices: list[float] | None = None, symbol: str | None = None) -> float | None:
        """Calcula EMA (Exponential Moving Average).
        Si `prices` es None, usa `self.price_history[symbol]`.
        """
        if prices is None:
            prices = self.price_history.get(symbol, []) if symbol else self.price_history.get("BTC/USDT", [])
            
        if len(prices) < period:
            return None
            
        # Usar últimos N precios
        recent = prices[-period:]
        
        # SMA inicial
        sma = sum(recent) / period
        
        # Multiplicador EMA
        multiplier = 2 / (period + 1)
        
        # Calcular EMA
        ema = sma
        for price in recent[1:]:
            ema = (price - ema) * multiplier + ema
            
        return ema

    def _calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9, symbol: str | None = None) -> tuple[float, float, float] | None:
        """Calcula MACD line, signal line, y histogram."""
        ph = self.price_history.get(symbol, []) if symbol else self.price_history.get("BTC/USDT", [])
        if len(ph) < slow:
            return None
            
        # EMA rápida y lenta
        ema_fast = self._calculate_ema(fast, prices=ph)
        ema_slow = self._calculate_ema(slow, prices=ph)
        
        if ema_fast is None or ema_slow is None:
            return None
            
        # MACD line = EMA rápida - EMA lenta
        macd_line = ema_fast - ema_slow
        
        # Signal line = EMA de 9 del MACD line
        # (Simplificación: usar promedio móvil simple por ahora)
        signal_line = macd_line * 0.9  # Aproximación simple
        
        # Histogram = MACD - Signal
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def _detect_candlestick_patterns(self, symbol: str | None = None) -> dict[str, bool]:
        """Detecta patrones básicos de velas japonesas."""
        patterns = {
            "hammer": False,
            "shooting_star": False,
            "bullish_engulfing": False,
            "bearish_engulfing": False,
            "doji": False
        }
        
        candles = self.candles.get(symbol, []) if symbol else self.candles.get("BTC/USDT", [])
        if len(candles) < 2:
            return patterns
            
        current = candles[-1]
        previous = candles[-2] if len(candles) >= 2 else None
        
        if not previous:
            return patterns
            
        # Calcular cuerpo y sombras
        body = abs(current.close - current.open)
        upper_shadow = current.high - max(current.open, current.close)
        lower_shadow = min(current.open, current.close) - current.low
        total_range = current.high - current.low
        
        # Patrón Hammer (alcista): cuerpo pequeño arriba, sombra inferior larga
        if (lower_shadow > body * 2 and 
            upper_shadow < body * 0.5 and 
            current.close > current.open):
            patterns["hammer"] = True
            
        # Patrón Shooting Star (bajista): cuerpo pequeño abajo, sombra superior larga
        if (upper_shadow > body * 2 and 
            lower_shadow < body * 0.5 and 
            current.close < current.open):
            patterns["shooting_star"] = True
            
        # Patrón Bullish Engulfing: vela verde grande envuelve vela roja anterior
        if (previous.close < previous.open and  # Anterior bajista
            current.close > current.open and     # Actual alcista
            current.open < previous.close and    # Abre por debajo
            current.close > previous.open):      # Cierra por encima
            patterns["bullish_engulfing"] = True
            
        # Patrón Bearish Engulfing: vela roja grande envuelve vela verde anterior
        if (previous.close > previous.open and  # Anterior alcista
            current.close < current.open and     # Actual bajista
            current.open > previous.close and    # Abre por encima
            current.close < previous.open):      # Cierra por debajo
            patterns["bearish_engulfing"] = True
            
        # Patrón Doji: apertura ≈ cierre (indecisión)
        if body < total_range * 0.1 and total_range > 0:
            patterns["doji"] = True
            
        return patterns

    def register_trade(self, side: str, price: float, size: float, result: dict, source: str = "manual", symbol: str = "BTC/USDT"):
        """
        Registra un trade manual o externo en la estrategia.
        
        Args:
            source: "auto" for EL GATO, "manual" for user trades
            symbol: Trading symbol (format: BTC/USDT)
        """
        import time
        from datetime import datetime
        
        # Inicializar símbolo si es necesario
        self._init_symbol(symbol)
        
        self.position_id += 1
        
        trade = {
            "id": result.get("id", f"T{int(time.time())}"),
            "position_id": self.position_id,
            "symbol": symbol,
            "side": side.upper(),
            "price": price,
            "size": size,
            "time": int(time.time()),
            "status": result.get("status", "FILLED"),
            "source": source,  # 🐱 auto or 👤 manual
            "result": result  # Store full result for debugging
        }
        
        # Update position tracking per symbol
        current_pos = self.positions.get(symbol, 0.0)
        current_entry = self.entry_prices.get(symbol, 0.0)
        
        if side.upper() == "BUY":
            if current_pos == 0:
                self.entry_prices[symbol] = price
            else:
                # Average down/up
                total_cost = (current_pos * current_entry) + (size * price)
                new_pos = current_pos + size
                self.entry_prices[symbol] = total_cost / new_pos if new_pos != 0 else price
            self.positions[symbol] = current_pos + size
            self.last_trade_time[symbol] = time.time()
            
            # 🆕 SINCRONIZACIÓN: Registrar en TradeTracker para tabla horizontal
            self.trade_tracker.open_trade(
                size=size,
                entry_price=price,
                side="LONG",
                symbol=symbol,
                trade_id=trade["id"],
                source=source  # 👤 manual or 🐱 auto
            )
            
        elif side.upper() == "SELL":
            if current_pos > 0:
                # Close or reduce position
                exit_pnl = (price - current_entry) * min(size, current_pos)
                self.realized_pnl += exit_pnl
                trade["pnl"] = exit_pnl
                self.positions[symbol] = current_pos - size
                if self.positions[symbol] <= 0:
                    self.positions[symbol] = 0
                    self.entry_prices[symbol] = 0
                    
                # 🆕 SINCRONIZACIÓN: Cerrar trades en TradeTracker para tabla horizontal
                # CRITICAL: Iterar sobre copia de la lista para evitar problemas de modificación
                closed_count = 0
                for t in list(self.trade_tracker.trades):  # ← list() crea una copia
                    if t['symbol'] == symbol and t['status'] == 'OPEN':
                        print(f"[SYNC] Cerrando trade #{t['id']} en TradeTracker @ ${price:.2f}")
                        self.trade_tracker.close_trade(t['id'], price)
                        closed_count += 1
                
                if closed_count > 0:
                    print(f"[SYNC] ✅ {closed_count} trade(s) cerrado(s) en tabla horizontal")
                else:
                    print(f"[SYNC] ⚠️ No se encontraron trades abiertos para {symbol} en TradeTracker")
            else:
                # Opening a SHORT position
                self.positions[symbol] = current_pos - size  # Negative position
                self.entry_prices[symbol] = price
                
                # 🆕 SINCRONIZACIÓN: Registrar SHORT en TradeTracker
                self.trade_tracker.open_trade(
                    size=size,
                    entry_price=price,
                    side="SHORT",
                    symbol=symbol,
                    trade_id=trade["id"],
                    source=source  # 👤 manual or 🐱 auto
                )
                
            self.last_trade_time[symbol] = time.time()
        
        self.trades.append(trade)       
        # Keep only last 100 trades to prevent memory issues
        if len(self.trades) > 100:
            self.trades = self.trades[-100:]
        
        # 🧠 Update Memory (if enabled)
        if self.memory_enabled:
            # Placeholder for trend_dir and volatility_level
            trend_dir = "neutral" 
            volatility_level = "normal"
            # Contexto de mercado
            context = MarketContext(
                trend=trend_dir,
                volatility=volatility_level,
            )
            # Removed legacy add_trade call to prevent AttributeError
        
        icon = "🐱" if source == "auto" else "👤"
        print(f"[TRADE REGISTERED] {icon} {source.upper()} {side.upper()} {size} {symbol} @ ${price:.2f} | Status: {result.get('status', 'N/A')}")
        return trade

    def check_trade(
        self,
        side: str,
        price: float,
    ) -> tuple[bool, float, float, float]:
        """
        Evalúa si un trade respeta los límites y devuelve (OK, size, SL, TP).
        """
        cfg = self.config
        max_risk_amount = cfg.capital_virtual * cfg.risk_per_trade
        loss_per_unit = price * cfg.stop_loss_pct

        if loss_per_unit <= 0:
            return False, 0, 0, 0

        size = max_risk_amount / loss_per_unit
        sl = price * (1 - cfg.stop_loss_pct)
        tp = price * (1 + cfg.take_profit_pct)

        if side not in {"BUY", "SELL"}:
            return False, 0, 0, 0

        return True, size, sl, tp


