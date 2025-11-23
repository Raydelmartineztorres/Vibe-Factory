"""
Motor de gestión de riesgo y position sizing con auto-trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from memory import MarketContext, TradeMemory


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
    """Implementa los cálculos clave de exposición y auto-trading."""

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
        
        self.memory = TradeMemory()
        
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
        
        # Verificar si el trading está habilitado
        try:
            from api import _trading_enabled
            if not _trading_enabled:
                return  # Trading pausado
        except:
            pass  # Si no se puede importar, continuar normalmente
            
        # Guardar historial de precios
        self.price_history.append(price)
        if len(self.price_history) > 20:
            self.price_history.pop(0)
        
        # Necesitamos al menos 10 precios para detectar tendencias
        if len(self.price_history) < 10:
            return
        
        # Evitar trades muy frecuentes (mínimo 5 segundos entre trades)
        current_time = time.time()
        if current_time - self.last_trade_time < 5:
            return
        
        
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
        
        # Consultar memoria
        if self.memory.should_avoid(current_context):
            return # Evitar trade por malas experiencias pasadas
            
        # --- RSI CHECK ---
        rsi = self._calculate_rsi(period=14)
        # Si no hay suficientes datos para RSI, asumimos neutral (50)
        current_rsi = rsi if rsi is not None else 50.0
        
        # --- ATR CHECK (Dynamic Risk) ---
        atr = self._calculate_atr(period=14)
        current_atr = atr if atr is not None else 50.0 # Fallback por si no hay velas suficientes
        
        # --- VOLUME CHECK ---
        avg_vol = self._calculate_avg_volume(period=20)
        current_vol = self.current_candle.volume if self.current_candle else 0
        # Si no hay historial, asumimos volumen OK. Si hay, pedimos al menos 50% del promedio.
        volume_ok = True
        if avg_vol > 0 and current_vol < (avg_vol * 0.5):
            volume_ok = False
            
        # --- FIN CHECKS ---

        # Señal alcista: precio subiendo (0.02%) Y RSI < 70 Y Volumen OK
        if recent_avg > older_avg * 1.0002 and self.position <= 0 and current_rsi < 70 and volume_ok:
            print(f"[AUTO-TRADE] 🟢 SEÑAL DE COMPRA detectada @ ${price:.2f} (RSI: {current_rsi:.1f}, ATR: {current_atr:.2f})")
            
            # Stop Loss Dinámico: Precio - 2 * ATR
            dynamic_sl = price - (2 * current_atr)
            
            order = {
                "symbol": "BTC_USD",
                "side": "BUY",
                "size": 0.001,
                "stop_loss": dynamic_sl,
                "take_profit": None
            }
            result = await execute_order(order, mode="demo")
            
            # Si teníamos short (posición negativa), calcular PnL al cerrar
            if self.position < 0:
                pnl = (self.entry_price - price) * abs(self.position)
                self.realized_pnl += pnl
                print(f"[PNL] Short cerrado. PnL: ${pnl:.2f}")
                
                # GUARDAR EXPERIENCIA
                self.memory.add_experience(current_context, pnl)
                
                self.position = 0
                self.entry_price = 0
            
            # Abrir Long
            else:
                self.position += 0.001
                self.entry_price = price
                
            self.last_trade_time = current_time
            self.trades.append({
                "id": result.get("id", f"auto_{current_time}"),
                "time": current_time,
                "side": "BUY",
                "price": price,
                "size": 0.001,
                "result": result,
                "source": "AUTO"
            })
            print(f"[AUTO-TRADE] Orden ejecutada: {result.get('id', 'N/A')} | SL: {dynamic_sl:.2f}")
        
        # Señal bajista: precio cayendo (0.02%) Y RSI > 30 Y Volumen OK
        elif recent_avg < older_avg * 0.9998 and self.position > 0 and current_rsi > 30 and volume_ok:
            print(f"[AUTO-TRADE] 🔴 SEÑAL DE VENTA detectada @ ${price:.2f} (RSI: {current_rsi:.1f}, ATR: {current_atr:.2f})")
            
            # Stop Loss Dinámico: Precio + 2 * ATR
            dynamic_sl = price + (2 * current_atr)
            
            order = {
                "symbol": "BTC_USD",
                "side": "SELL",
                "size": 0.001,
                "stop_loss": dynamic_sl,
                "take_profit": None
            }
            result = await execute_order(order, mode="demo")
            
            # Si teníamos Long (posición positiva), calcular PnL al cerrar
            if self.position > 0:
                pnl = (price - self.entry_price) * abs(self.position)
                self.realized_pnl += pnl
                print(f"[PNL] Long cerrado. PnL: ${pnl:.2f}")
                
                # GUARDAR EXPERIENCIA
                self.memory.add_experience(current_context, pnl)
                
                self.position = 0
                self.entry_price = 0
            
            # Abrir Short
            else:
                self.position -= 0.001
                self.entry_price = price

            self.last_trade_time = current_time
            self.trades.append({
                "id": result.get("id", f"auto_{current_time}"),
                "time": current_time,
                "side": "SELL",
                "price": price,
                "size": 0.001,
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

    def register_trade(self, side: str, price: float, size: float, result: dict) -> None:
        """Registra un trade manual o externo en la estrategia."""
        import time
        current_time = time.time()
        
        # Contexto manual (aproximado)
        ctx = MarketContext(trend="MANUAL", volatility="MANUAL") 
        
        # Actualizar posición y PnL
        if side == "BUY":
            # Si teníamos short, cerrar y realizar PnL
            if self.position < 0:
                # Asumimos cierre total por simplicidad en demo
                pnl = (self.entry_price - price) * abs(self.position)
                self.realized_pnl += pnl
                self.memory.add_experience(ctx, pnl) # Guardar experiencia manual
                self.position = 0
                self.entry_price = 0
            else:
                # Promediar precio de entrada si ya teníamos long (simple average)
                total_cost = (self.position * self.entry_price) + (size * price)
                self.position += size
                self.entry_price = total_cost / self.position if self.position > 0 else price
                
        elif side == "SELL":
            # Si teníamos long, cerrar y realizar PnL
            if self.position > 0:
                pnl = (price - self.entry_price) * abs(self.position) # Asumiendo cierre parcial o total
                # Si cerramos todo
                if size >= self.position:
                    self.realized_pnl += pnl
                    self.memory.add_experience(ctx, pnl) # Guardar experiencia manual
                    self.position = 0
                    self.entry_price = 0
                else:
                    # Cierre parcial
                    self.realized_pnl += (price - self.entry_price) * size
                    self.position -= size
            else:
                # Abrir short
                total_cost = (abs(self.position) * self.entry_price) + (size * price)
                self.position -= size
                self.entry_price = total_cost / abs(self.position) if abs(self.position) > 0 else price

        self.last_trade_time = current_time
        self.trades.append({
            "id": result.get("id", f"manual_{current_time}"),
            "time": current_time,
            "side": side,
            "price": price,
            "size": size,
            "result": result,
            "source": "MANUAL"
        })

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


