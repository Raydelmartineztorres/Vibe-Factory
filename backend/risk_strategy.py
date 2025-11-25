"""
Motor de gestión de riesgo y position sizing con auto-trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, Protocol
from datetime import datetime
import random

from memory import MarketContext, TradeMemory
from strategy_profiles import get_strategy_manager


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
        self.price_history = []
        self.candles: list[Candle] = []
        self.current_candle: Candle | None = None
        self.position = 0.0
        self.entry_price = 0.0  # Precio promedio de entrada
        self.realized_pnl = 0.0 # Ganancia/Pérdida acumulada cerrada
        self.unrealized_pnl = 0.0 # Ganancia/Pérdida flotante
        self.trades = []
        self.last_trade_time = 0
        self.last_price = 0.0
        
        # 🧠 INTELIGENCIA ADAPTATIVA
        from ml_predictor import MLPredictor
        from market_timing import MarketTiming
        from sentiment_manager import SentimentManager
        from risk_manager import RiskManager
        
        # ML y Memory config (por defecto activados)
        self.ml_enabled = False
        self.memory_enabled = True
        
        # Inicializar componentes
        self.ml_predictor = MLPredictor()
        self.market_timing = MarketTiming()
        self.sentiment_manager = SentimentManager()
        self.risk_manager = RiskManager(initial_capital=self.config.capital_virtual)
        self.memory = TradeMemory()
        self.last_prediction = None
        self.position_id = 0  # Contador para IDs de posición
        
        # Sistema de estrategias múltiples
        self.strategy_manager = get_strategy_manager()
        print(f"[STRATEGY] ✅ Inicializado con estrategia: {self.strategy_manager.active_strategy}")
        
    def _fill_missing_candles(self, from_time: int, to_time: int, last_price: float):
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
            self.candles.append(filler_candle)
            current_time += 5
    
    def _update_candle(self, price: float, timestamp: float):
        """Agrega ticks a la vela actual (5 segundos)."""
        # Redondear al múltiplo de 5 segundos más cercano (floor)
        ts_sec = int(timestamp)
        candle_time = ts_sec - (ts_sec % 5)
        
        # Simular volumen aleatorio por tick
        import random
        tick_vol = random.uniform(1.0, 10.0) 
        
        if self.current_candle is None:
            self.current_candle = Candle(
                time=candle_time, open=price, high=price, low=price, close=price, volume=tick_vol
            )
        elif self.current_candle.time == candle_time:
            # Actualizar vela existente
            self.current_candle.high = max(self.current_candle.high, price)
            self.current_candle.low = min(self.current_candle.low, price)
            self.current_candle.close = price
            self.current_candle.volume += tick_vol
        else:
            # ⭐ DETECTAR Y RELLENAR GAPS
            last_candle_time = self.current_candle.time
            time_gap = candle_time - last_candle_time
            
            # Si hay gap mayor a 5 segundos, rellenar
            if time_gap > 5:
                last_close = self.current_candle.close
                self._fill_missing_candles(last_candle_time, candle_time, last_close)
            
            # Cerrar vela anterior y abrir nueva
            self.candles.append(self.current_candle)
            # Mantener solo últimas 1000 velas
            if len(self.candles) > 1000:
                self.candles.pop(0)
                
            self.current_candle = Candle(
                time=candle_time, open=price, high=price, low=price, close=price, volume=tick_vol
            )

    def _calculate_avg_volume(self, period: int = 20) -> float:
        """Calcula el volumen promedio de las últimas N velas."""
        if not self.candles:
            return 0.0
        recent_candles = self.candles[-period:]
        total_vol = sum(c.volume for c in recent_candles)
        return total_vol / len(recent_candles)

    async def on_tick(self, payload: dict) -> None:
        """Hook llamado por `data_collector` en cada tick live."""
        import time
        from broker_api_handler import execute_order
        
        symbol = payload.get("symbol")
        price = payload.get("price")
        
        if not price:
            return
            
        self.last_price = price
        self._update_candle(price, time.time())
        
        # Actualizar PnL No Realizado
        if self.position > 0:
            self.unrealized_pnl = (price - self.entry_price) * self.position
        elif self.position < 0:
            self.unrealized_pnl = (self.entry_price - price) * abs(self.position)
        else:
            self.unrealized_pnl = 0.0
        
        # 🛡️ ACTUALIZAR BALANCE PARA DRAWDOWN PROTECTION
        total_balance = self.config.capital_virtual + self.realized_pnl + self.unrealized_pnl
        self.risk_manager.update_balance(total_balance)
        
        # Verificar si el trading está habilitado
        try:
            from api import _trading_enabled
            if not _trading_enabled:
                return  # Trading pausado
        except:
            pass  # Si no se puede importar, continuar normalmente
            
        # Guardar historial de precios
        self.price_history.append(price)
        if len(self.price_history) > 100:  # Aumentado a 100 para ML
            self.price_history.pop(0)
        
        # Necesitamos al menos 26 precios para MACD y otros indicadores
        if len(self.price_history) < 26:
            return
        
        # Evitar trades muy frecuentes (mínimo 5 segundos entre trades)
        current_time = time.time()
        if current_time - self.last_trade_time < 5:
            return
        
        # 🧠 === INTELIGENCIA ADAPTATIVA ===
        
        # 1. MARKET TIMING: Registrar volumen y obtener sesión actual
        current_vol = self.current_candle.volume if self.current_candle else 0
        self.market_timing.record_volume(current_vol, current_time)
        session = self.market_timing.get_current_session()
        
        print(f"[ADAPTIVE] Sesión: {session.name} | Agresividad: {session.aggressiveness:.2f}x | Volumen: {current_vol:.1f}")
        
        # 2. CALCULAR INDICADORES TÉCNICOS
        rsi = self._calculate_rsi(period=14)
        current_rsi = rsi if rsi is not None else 50.0
        
        atr = self._calculate_atr(period=14)
        current_atr = atr if atr is not None else 50.0
        
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
                    prev_price  # El precio "futuro" que ocurrió
                )
            
            # Predecir próximo precio
            predicted_price = self.ml_predictor.predict(
                self.price_history, 
                current_rsi, 
                current_atr, 
                current_vol
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
        
        # 4. AJUSTAR PARÁMETROS SEGÚN MARKET TIMING
        base_params = {
            'rsi_buy_threshold': 70,
            'rsi_sell_threshold': 30,
            'volume_multiplier': 0.5
        }
        adjusted_params = self.market_timing.adjust_parameters(base_params)
        
        print(f"[ADAPTIVE] RSI Thresholds: BUY<{adjusted_params['rsi_buy_threshold']}, SELL>{adjusted_params['rsi_sell_threshold']}")
        
        # === FIN INTELIGENCIA ADAPTATIVA ===
        

        # --- MEMORY CHECK ---
        
        # Determinar contexto actual
        trend_dir = "FLAT"
        recent_avg = sum(self.price_history[-5:]) / 5
        older_avg = sum(self.price_history[-10:-5]) / 5
        if recent_avg > older_avg * 1.0001: trend_dir = "UP"
        elif recent_avg < older_avg * 0.9999: trend_dir = "DOWN"
        
        volatility_level = "NORMAL"
        # Simple volatilidad basada en rango de precios recientes
        recent_range = max(self.price_history[-10:]) - min(self.price_history[-10:])
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
        if self.position != 0:
            current_pos_id = f"pos_{self.position_id}"
            atr_for_trailing = self._calculate_atr(period=14) or 50.0
            
            # Actualizar trailing stop
            new_sl = self.risk_manager.update_trailing_stop(current_pos_id, price, atr_for_trailing)
            
            # Verificar si el trailing stop fue alcanzado
            if self.risk_manager.check_trailing_stop_hit(current_pos_id, price):
                print(f"[RISK] 🛑 Cerrando posición por Trailing Stop")
                
                # Cerrar posición
                if self.position > 0:
                    # Cerrar LONG
                    pnl = (price - self.entry_price) * self.position
                    order = {
                        "symbol": "BTC_USD",
                        "side": "SELL",
                        "size": self.position,
                        "stop_loss": None,
                        "take_profit": None
                    }
                else:
                    # Cerrar SHORT
                    pnl = (self.entry_price - price) * abs(self.position)
                    order = {
                        "symbol": "BTC_USD",
                        "side": "BUY",
                        "size": abs(self.position),
                        "stop_loss": None,
                        "take_profit": None
                    }
                
                result = await execute_order(order, mode="demo")
                self.realized_pnl += pnl
                print(f"[PNL] Trailing Stop cerrado. PnL: ${pnl:.2f}")
                
                # Guardar experiencia
                rsi_val = self._calculate_rsi(period=14) or 50.0
                self.memory.add_experience(current_context, pnl, rsi=rsi_val)
                
                # Limpiar
                self.risk_manager.remove_trailing_stop(current_pos_id)
                self.risk_manager.remove_pyramid(current_pos_id)
                self.position = 0
                self.entry_price = 0
                
                return  # No generar nuevas señales este tick
            
        # --- RSI CHECK ---
        rsi = self._calculate_rsi(period=14)
        current_rsi = rsi if rsi is not None else 50.0
        
        # --- ATR CHECK (Dynamic Risk) ---
        atr = self._calculate_atr(period=14)
        current_atr = atr if atr is not None else 50.0
        
        # --- VOLUME CHECK ---
        avg_vol = self._calculate_avg_volume(period=20)
        current_vol = self.current_candle.volume if self.current_candle else 0
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

        # ===== SEÑAL ALCISTA (COMPRA) =====
        # Condiciones adaptativas: EMA + MACD + RSI ajustado + Volumen + ML (opcional)
        ml_confirms_buy = (ml_signal == "BUY") if self.ml_enabled else True
        
        # 🛡️ RISK MANAGER: Verificar drawdown
        if not self.risk_manager.should_trade():
            return  # No tradear si drawdown es excesivo
        
        if (self.position <= 0 and 
            ema_bullish and 
            macd_bullish and 
            current_rsi < adjusted_params['rsi_buy_threshold'] and  # Umbral adaptivo
            volume_ok and
            ml_confirms_buy):  # Confirmación ML
            
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
            
            order = {
                "symbol": "BTC_USD",
                "side": "BUY",
                "size": position_size,  # ⭐ Tamaño dinámico
                "stop_loss": dynamic_sl,
                "take_profit": None
            }
            result = await execute_order(order, mode="demo")
            
            # Si teníamos short (posición negativa), calcular PnL al cerrar
            if self.position < 0:
                pnl = (self.entry_price - price) * abs(self.position)
                self.realized_pnl += pnl
                print(f"[PNL] Short cerrado. PnL: ${pnl:.2f}")
                
                # GUARDAR EXPERIENCIA MEJORADA
                pattern_name = detected_patterns[0] if detected_patterns else None
                self.memory.add_experience(current_context, pnl, rsi=current_rsi, pattern=pattern_name, ml_signal=ml_signal)
                
                # 🛡️ Remover trailing stop del short
                self.risk_manager.remove_trailing_stop(f"pos_{self.position_id}")
                self.risk_manager.remove_pyramid(f"pos_{self.position_id}")
                
                self.position = 0
                self.entry_price = 0
            
            # Abrir Long o Pyramiding
            self.position_id += 1
            current_pos_id = f"pos_{self.position_id}"
            
            if self.position == 0:
                # Nueva posición
                self.position += position_size
                self.entry_price = price
                
                # 🛡️ Inicializar trailing stop
                self.risk_manager.init_trailing_stop(current_pos_id, price, dynamic_sl, "LONG")
                
                # 🛡️ Inicializar pyramid tracking
                self.risk_manager.init_pyramid(current_pos_id, price, position_size)
                
            else:
                # Potential pyramiding
                prev_pos_id = f"pos_{self.position_id - 1}"
                if self.risk_manager.should_pyramid(prev_pos_id, price, "LONG", ml_signal):
                    pyramid_size = self.risk_manager.add_pyramid_entry(prev_pos_id, price)
                    self.position += pyramid_size
                    # Actualizar precio promedio
                    self.entry_price = ((self.entry_price * (self.position - pyramid_size)) + 
                                       (price * pyramid_size)) / self.position
                    print(f"[🔺 PYRAMID] Posición aumentada: {self.position:.6f} BTC | Avg: ${self.entry_price:.2f}")
            
            self.last_trade_time = current_time
            self.trades.append({
                "id": result.get("id", f"auto_{current_time}"),
                "time": current_time,
                "side": "BUY",
                "price": price,
                "size": position_size,
                "result": result,
                "source": "AUTO"
            })
            print(f"[AUTO-TRADE] Orden ejecutada: {result.get('id', 'N/A')} | SL: {dynamic_sl:.2f}")
        
        
        # ===== SEÑAL BAJISTA (VENTA) =====
        # Condiciones adaptativas: EMA + MACD + RSI ajustado + Volumen + ML (opcional)
        ml_confirms_sell = (ml_signal == "SELL") if self.ml_enabled else True
        
        if (self.position >= 0 and 
              ema_bearish and 
              macd_bearish and 
              current_rsi > adjusted_params['rsi_sell_threshold'] and  # Umbral adaptivo
              volume_ok and
              ml_confirms_sell):  # Confirmación ML
            
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
            
            position_size = self.risk_manager.calculate_position_size(
                ml_signal=ml_signal,
                memory_winrate=context_winrate,
                pattern_detected=has_pattern,
                volatility_ratio=vol_ratio,
                multiple_confirmations=multiple_confirm
            )
            
            # Stop Loss Dinámico ajustado por sesión
            sl_multiplier = 2.0 / session.aggressiveness
            dynamic_sl = price + (sl_multiplier * current_atr)
            
            order = {
                "symbol": "BTC_USD",
                "side": "SELL",
                "size": position_size,  # ⭐ Tamaño dinámico
                "stop_loss": dynamic_sl,
                "take_profit": None
            }
            result = await execute_order(order, mode="demo")
            
            # Si teníamos long (posición positiva), calcular PnL al cerrar
            if self.position > 0:
                pnl = (price - self.entry_price) * self.position
                self.realized_pnl += pnl
                print(f"[PNL] Long cerrado. PnL: ${pnl:.2f}")
                
                # GUARDAR EXPERIENCIA MEJORADA
                pattern_name = detected_patterns[0] if detected_patterns else None
                self.memory.add_experience(current_context, pnl, rsi=current_rsi, pattern=pattern_name, ml_signal=ml_signal)
                
                # 🛡️ Remover trailing stop del long
                self.risk_manager.remove_trailing_stop(f"pos_{self.position_id}")
                self.risk_manager.remove_pyramid(f"pos_{self.position_id}")
                
                self.position = 0
                self.entry_price = 0
            
            # Abrir Short o Pyramiding
            self.position_id += 1
            current_pos_id = f"pos_{self.position_id}"
            
            if self.position == 0:
                # Nueva posición
                self.position -= position_size
                self.entry_price = price
                
                # 🛡️ Inicializar trailing stop
                self.risk_manager.init_trailing_stop(current_pos_id, price, dynamic_sl, "SHORT")
                
                # 🛡️ Inicializar pyramid tracking
                self.risk_manager.init_pyramid(current_pos_id, price, position_size)
            else:
                # Potential pyramiding
                prev_pos_id = f"pos_{self.position_id - 1}"
                if self.risk_manager.should_pyramid(prev_pos_id, price, "SHORT", ml_signal):
                    pyramid_size = self.risk_manager.add_pyramid_entry(prev_pos_id, price)
                    self.position -= pyramid_size
                    # Actualizar precio promedio
                    self.entry_price = ((self.entry_price * (abs(self.position) - pyramid_size)) + 
                                       (price * pyramid_size)) / abs(self.position)
                    print(f"[🔺 PYRAMID] Posición aumentada: {abs(self.position):.6f} BTC | Avg: ${self.entry_price:.2f}")
            
            self.last_trade_time = current_time
            self.trades.append({
                "id": result.get("id", f"auto_{current_time}"),
                "time": current_time,
                "side": "SELL",
                "price": price,
                "size": position_size,
                "result": result,
                "source": "AUTO"
            })
            print(f"[AUTO-TRADE] Orden ejecutada: {result.get('id', 'N/A')} | SL: {dynamic_sl:.2f}")

    def _calculate_atr(self, period: int = 14) -> float | None:
        """Calcula el ATR basado en las velas."""
        if len(self.candles) < period + 1:
            return None
            
        # Usar las últimas N velas
        candles = self.candles[-(period+1):]
        tr_list = []
        
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i-1].close
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
            
        return sum(tr_list) / period

    def _calculate_rsi(self, period: int = 14) -> float | None:
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

    def _calculate_ema(self, period: int, prices: list[float] | None = None) -> float | None:
        """Calcula EMA (Exponential Moving Average)."""
        if prices is None:
            prices = self.price_history
            
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

    def _calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float] | None:
        """Calcula MACD line, signal line, y histogram."""
        if len(self.price_history) < slow:
            return None
            
        # EMA rápida y lenta
        ema_fast = self._calculate_ema(fast)
        ema_slow = self._calculate_ema(slow)
        
        if ema_fast is None or ema_slow is None:
            return None
            
        # MACD line = EMA rápida - EMA lenta
        macd_line = ema_fast - ema_slow
        
        # Signal line = EMA de 9 del MACD line
        # (Simplificación: usar promedio móvil simple por ahora)
        # En producción, calcularíamos EMA del historial de MACD
        signal_line = macd_line * 0.9  # Aproximación simple
        
        # Histogram = MACD - Signal
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def _detect_candlestick_patterns(self) -> dict[str, bool]:
        """Detecta patrones básicos de velas japonesas."""
        patterns = {
            "hammer": False,
            "shooting_star": False,
            "bullish_engulfing": False,
            "bearish_engulfing": False,
            "doji": False
        }
        
        if len(self.candles) < 2:
            return patterns
            
        current = self.candles[-1]
        previous = self.candles[-2] if len(self.candles) >= 2 else None
        
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

    def register_trade(self, side: str, price: float, size: float, result: dict, source: str = "manual"):
        """
        Registra un trade manual o externo en la estrategia.
        
        Args:
            source: "auto" for EL GATO, "manual" for user trades
        """
        import time
        from datetime import datetime
        
        self.position_id += 1
        
        trade = {
            "id": result.get("id", f"T{int(time.time())}"),
            "position_id": self.position_id,
            "side": side.upper(),
            "price": price,
            "size": size,
            "time": int(time.time()),
            "status": result.get("status", "FILLED"),
            "source": source,  # 🐱 auto or 👤 manual
        }
        
        # Update position tracking
        if side.upper() == "BUY":
            if self.position == 0:
                self.entry_price = price
            else:
                # Average down/up
                total_cost = (self.position * self.entry_price) + (size * price)
                self.position += size
                self.entry_price = total_cost / self.position if self.position != 0 else price
            self.position += size
            self.last_trade_time = time.time()
        elif side.upper() == "SELL":
            if self.position > 0:
                # Close or reduce position
                exit_pnl = (price - self.entry_price) * min(size, self.position)
                self.realized_pnl += exit_pnl
                trade["pnl"] = exit_pnl
                self.position -= size
                if self.position <= 0:
                    self.position = 0
                    self.entry_price = 0
            self.last_trade_time = time.time()
        
        self.trades.append(trade)
        
        # 🧠 Update Memory
        if self.memory_enabled:
            # Placeholder for trend_dir and volatility_level, as they are not defined in the provided context.
            # Assuming they would be calculated or retrieved elsewhere in a full implementation.
            trend_dir = "neutral" # Example placeholder
            volatility_level = "medium" # Example placeholder

            context = MarketContext(
                trend=trend_dir,
                volatility=volatility_level,
                day_of_week=datetime.now().strftime("%A"),
            )
            self.memory.add_trade(side, price, True, context)
        
        icon = "🐱" if source == "auto" else "👤"
        print(f"[STRATEGY] {icon} Trade registered: {side} {size} @ ${price:.2f}")
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


