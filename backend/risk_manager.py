"""
Módulo de Gestión Avanzada de Riesgo.

Implementa:
- Trailing Stop-Loss dinámico
- Position Sizing inteligente
- Pyramiding (agregar a posiciones ganadoras)
- Max Drawdown Protection
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timezone

@dataclass
class AdvancedRiskConfig:
    """Configuración del sistema de gestión de riesgo."""
    
    # Trailing Stop-Loss
    trailing_enabled: bool = True
    trailing_atr_multiplier: float = 1.0  # 🔥 Tighter stops (was 1.5)
    trailing_activation_threshold: float = 0.001  # 🔥 Activate at 0.1% profit (was 0.3%)
    
    # Position Sizing
    base_position_size: float = 0.001
    min_position_size: float = 0.0005
    max_position_size: float = 0.003
    
    # Pyramiding
    pyramid_enabled: bool = True
    pyramid_max_levels: int = 3
    pyramid_profit_threshold: float = 0.005  # 0.5% profit mínimo
    pyramid_size_reduction: float = 0.7  # Cada nivel es 70% del anterior
    
    # Max Drawdown
    max_drawdown_alert: float = 0.02   # 2%
    max_drawdown_caution: float = 0.04  # 4%
    max_drawdown_danger: float = 0.06   # 6%
    max_drawdown_stop: float = 0.10     # 10%
    
    # 🆕 Trading Fees (Exchange Commissions)
    maker_fee: float = 0.001  # 0.1% (Binance maker fee)
    taker_fee: float = 0.001  # 0.1% (Binance taker fee - más común)
    min_profit_multiplier: float = 1.1  # 🔥 AGGRESSIVE: Ganancia mínima = fees × 1.1


@dataclass
class TrailingStop:
    """Información de un trailing stop-loss."""
    initial_price: float
    initial_sl: float
    current_sl: float
    best_price: float  # Mejor precio alcanzado
    direction: str  # "LONG" or "SHORT"
    activated: bool = False


@dataclass
class PyramidEntry:
    """Una entrada de pyramiding."""
    level: int
    entry_price: float
    size: float
    timestamp: float


class RiskManager:
    """Gestor avanzado de riesgo."""
    
    def __init__(self, config: AdvancedRiskConfig = None, initial_capital: float = 10000.0):
        self.config = config or AdvancedRiskConfig()
        
        # Trailing stops por posición
        self.trailing_stops: Dict[str, TrailingStop] = {}
        
        # Pyramid entries por posición
        self.pyramid_entries: Dict[str, List[PyramidEntry]] = {}
        
        # Drawdown tracking
        self.peak_balance: float = initial_capital
        self.current_balance: float = initial_capital
        self.daily_drawdown: float = 0.0
        self.drawdown_state: str = "NORMAL"  # NORMAL/ALERT/CAUTION/DANGER/STOP
        
        # Stats
        self.total_pyramids: int = 0
        self.total_trailing_stops_hit: int = 0
        
    # ========== TRAILING STOP-LOSS ==========
    
    def init_trailing_stop(self, position_id: str, entry_price: float, 
                          initial_sl: float, direction: str):
        """Inicializa un trailing stop para una posición."""
        self.trailing_stops[position_id] = TrailingStop(
            initial_price=entry_price,
            initial_sl=initial_sl,
            current_sl=initial_sl,
            best_price=entry_price,
            direction=direction,
            activated=False
        )
        print(f"[RISK] Trailing SL inicializado: {direction} @ ${entry_price:.2f} | SL: ${initial_sl:.2f}")
    
    def update_trailing_stop(self, position_id: str, current_price: float, 
                            atr: float) -> Optional[float]:
        """
        Actualiza el trailing stop-loss si el precio es favorable.
        Retorna el nuevo SL si cambió, None si no.
        """
        if not self.config.trailing_enabled:
            return None
            
        if position_id not in self.trailing_stops:
            return None
            
        ts = self.trailing_stops[position_id]
        
        # Calcular profit actual
        if ts.direction == "LONG":
            profit_pct = (current_price - ts.initial_price) / ts.initial_price
            is_better_price = current_price > ts.best_price
        else:  # SHORT
            profit_pct = (ts.initial_price - current_price) / ts.initial_price
            is_better_price = current_price < ts.best_price
        
        # Activar trailing si alcanzamos threshold
        if not ts.activated and profit_pct >= self.config.trailing_activation_threshold:
            ts.activated = True
            print(f"[RISK] 🎯 Trailing SL ACTIVADO para {position_id} (profit: {profit_pct*100:.2f}%)")
        
        # Si no está activado, no hacer nada
        if not ts.activated:
            return None
        
        # Si tenemos nuevo mejor precio, actualizar SL
        if is_better_price:
            ts.best_price = current_price
            
            # Calcular nueva distancia SL basada en ATR
            trailing_distance = atr * self.config.trailing_atr_multiplier
            
            if ts.direction == "LONG":
                new_sl = current_price - trailing_distance
                # Solo mover SL hacia arriba (nunca hacia abajo)
                if new_sl > ts.current_sl:
                    old_sl = ts.current_sl
                    ts.current_sl = new_sl
                    print(f"[RISK] ⬆️ Trailing SL actualizado: ${old_sl:.2f} → ${new_sl:.2f} (+${new_sl-old_sl:.2f})")
                    return new_sl
            else:  # SHORT
                new_sl = current_price + trailing_distance
                # Solo mover SL hacia abajo (nunca hacia arriba)
                if new_sl < ts.current_sl:
                    old_sl = ts.current_sl
                    ts.current_sl = new_sl
                    print(f"[RISK] ⬇️ Trailing SL actualizado: ${old_sl:.2f} → ${new_sl:.2f} (-${old_sl-new_sl:.2f})")
                    return new_sl
        
        return None
    
    def check_trailing_stop_hit(self, position_id: str, current_price: float) -> bool:
        """Verifica si el trailing stop fue alcanzado."""
        if position_id not in self.trailing_stops:
            return False
            
        ts = self.trailing_stops[position_id]
        
        if ts.direction == "LONG" and current_price <= ts.current_sl:
            print(f"[RISK] 🛑 Trailing SL alcanzado (LONG): ${current_price:.2f} <= ${ts.current_sl:.2f}")
            self.total_trailing_stops_hit += 1
            return True
        elif ts.direction == "SHORT" and current_price >= ts.current_sl:
            print(f"[RISK] 🛑 Trailing SL alcanzado (SHORT): ${current_price:.2f} >= ${ts.current_sl:.2f}")
            self.total_trailing_stops_hit += 1
            return True
            
        return False
    
    def remove_trailing_stop(self, position_id: str):
        """Elimina un trailing stop."""
        if position_id in self.trailing_stops:
            del self.trailing_stops[position_id]
    
    # ========== POSITION SIZING ==========
    
    def calculate_position_size(self, ml_signal: str, memory_winrate: float, 
                              pattern_detected: bool, volatility_ratio: float,
                              multiple_confirmations: bool, singularity_multiplier: float = 1.0) -> float:
        """
        Calcula el tamaño de posición óptimo basado en factores de riesgo.
        
        Args:
            ml_signal: Señal del modelo ML (BUY/SELL/HOLD)
            memory_winrate: Win rate histórico en contexto similar (0.0-1.0)
            pattern_detected: Si se detectó un patrón de vela
            volatility_ratio: Ratio de volatilidad actual vs promedio
            multiple_confirmations: Si hay múltiples confirmaciones técnicas
            singularity_multiplier: Factor de escala basado en inteligencia (default 1.0)
        
        Returns:
            Tamaño de posición ajustado
        """
        base_size = self.config.base_position_size
        
        # 🧠 SINGULARITY SCALING: Aumentar límite máximo basado en inteligencia
        # Risk = Base * sqrt(Multiplier) para crecimiento seguro pero potente
        # Ejemplo: Multiplier 10x -> Risk 3.16x
        import math
        scaled_max_size = self.config.max_position_size * math.sqrt(singularity_multiplier)
        
        # === CONFIDENCE BOOSTS ===
        confidence_boost = 0.0
        
        # ML Signal boost
        if ml_signal in ["BUY", "SELL"]:
            confidence_boost += 0.30
            
        # Memory boost
        if memory_winrate > 0.60:
            memory_boost = (memory_winrate - 0.5) * 0.4  # Max +0.04 (40% de 0.1)
            confidence_boost += memory_boost
            
        # Pattern boost
        if pattern_detected:
            confidence_boost += 0.15
            
        # Multiple confirmations
        if multiple_confirmations:
            confidence_boost += 0.15
            
        # Volatility boost (baja volatilidad = más seguro)
        if volatility_ratio < 0.8:
            confidence_boost += 0.10
        
        # === RISK PENALTIES ===
        risk_penalty = 0.0
        
        # Memory penalty
        if memory_winrate < 0.40:
            memory_penalty = (0.5 - memory_winrate) * 0.8  # Max -0.08 (80% de 0.1)
            risk_penalty += memory_penalty
            
        # Volatility penalty
        if volatility_ratio > 1.2:
            risk_penalty += 0.20
            
        # Drawdown penalty
        if self.drawdown_state == "ALERT":
            risk_penalty += 0.30
        elif self.drawdown_state == "CAUTION":
            risk_penalty += 0.50
        elif self.drawdown_state == "DANGER":
            risk_penalty += 0.70
            
        # === FINAL CALCULATION ===
        # Size = Base * (1 + Boosts - Penalties)
        total_adjustment = 1.0 + confidence_boost - risk_penalty
        
        # Ensure adjustment is positive
        total_adjustment = max(0.1, total_adjustment)
        
        final_size = base_size * total_adjustment
        
        # Apply limits (usando scaled_max_size)
        final_size = max(self.config.min_position_size, min(final_size, scaled_max_size))
        
        # Round to 4 decimals
        final_size = round(final_size, 4)
        
        print(f"[RISK] Position Size: {final_size:.6f} BTC (base: {base_size:.6f}, multiplier: {total_adjustment:.2f}x)")
        print(f"  └─ Confidence: +{confidence_boost*100:.1f}% | Risk: -{risk_penalty*100:.1f}%")
        
        return final_size
    
    # ========== PYRAMIDING ==========
    
    def init_pyramid(self, position_id: str, entry_price: float, size: float):
        """Inicializa tracking de pyramiding para una posición."""
        self.pyramid_entries[position_id] = [
            PyramidEntry(
                level=1,
                entry_price=entry_price,
                size=size,
                timestamp=time.time()
            )
        ]
    
    def should_pyramid(self, position_id: str, current_price: float, 
                       direction: str, ml_signal: str = "NEUTRAL") -> bool:
        """
        Decide si debemos agregar a la posición (pyramid).
        
        Returns:
            True si debemos agregar, False si no
        """
        if not self.config.pyramid_enabled:
            return False
            
        if position_id not in self.pyramid_entries:
            return False
            
        entries = self.pyramid_entries[position_id]
        
        # Check 1: No exceder max levels
        if len(entries) >= self.config.pyramid_max_levels:
            return False
        
        # Check 2: Calcular profit promedio
        total_value = 0.0
        total_size = 0.0
        
        for entry in entries:
            total_value += entry.entry_price * entry.size
            total_size += entry.size
        
        avg_entry_price = total_value / total_size if total_size > 0 else 0
        
        if direction == "LONG":
            profit_pct = (current_price - avg_entry_price) / avg_entry_price
        else:  # SHORT
            profit_pct = (avg_entry_price - current_price) / avg_entry_price
        
        # Check 3: Profit suficiente?
        if profit_pct < self.config.pyramid_profit_threshold:
            return False
        
        # Check 4: ML debe confirmar (opcional pero recomendado)
        if direction == "LONG" and ml_signal == "SELL":
            return False
        if direction == "SHORT" and ml_signal == "BUY":
            return False
        
        print(f"[RISK] ✅ Pyramiding disponible: Nivel {len(entries)+1}/3 | Profit: {profit_pct*100:.2f}%")
        return True
    
    def add_pyramid_entry(self, position_id: str, entry_price: float) -> float:
        """
        Agrega una entrada de pyramid y retorna el tamaño a usar.
        
        Returns:
            Tamaño de la nueva entrada
        """
        if position_id not in self.pyramid_entries:
            return 0.0
            
        entries = self.pyramid_entries[position_id]
        level = len(entries) + 1
        
        # Calcular tamaño reducido
        first_entry_size = entries[0].size
        new_size = first_entry_size * (self.config.pyramid_size_reduction ** (level - 1))
        
        # Agregar entrada
        entries.append(PyramidEntry(
            level=level,
            entry_price=entry_price,
            size=new_size,
            timestamp=time.time()
        ))
        
        self.total_pyramids += 1
        
        print(f"[RISK] 🔺 Pyramid Entry #{level}: {new_size:.6f} BTC @ ${entry_price:.2f}")
        
        return new_size
    
    def remove_pyramid(self, position_id: str):
        """Elimina tracking de pyramid."""
        if position_id in self.pyramid_entries:
            del self.pyramid_entries[position_id]
    
    # ========== MAX DRAWDOWN ==========
    
    def update_balance(self, current_balance: float):
        """Actualiza el balance y calcula drawdown."""
        self.current_balance = current_balance
        
        # Actualizar peak
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Calcular drawdown
        if self.peak_balance > 0:
            self.daily_drawdown = (self.peak_balance - current_balance) / self.peak_balance
        else:
            self.daily_drawdown = 0.0
        
        # Actualizar estado
        old_state = self.drawdown_state
        
        if self.daily_drawdown >= self.config.max_drawdown_stop:
            self.drawdown_state = "STOP"
        elif self.daily_drawdown >= self.config.max_drawdown_danger:
            self.drawdown_state = "DANGER"
        elif self.daily_drawdown >= self.config.max_drawdown_caution:
            self.drawdown_state = "CAUTION"
        elif self.daily_drawdown >= self.config.max_drawdown_alert:
            self.drawdown_state = "ALERT"
        else:
            self.drawdown_state = "NORMAL"
        
        # Log si cambió estado
        if old_state != self.drawdown_state:
            emoji = {"NORMAL": "🟢", "ALERT": "🟡", "CAUTION": "🟠", "DANGER": "🔴", "STOP": "⛔"}
            print(f"\n[RISK] {emoji[self.drawdown_state]} DRAWDOWN STATE: {old_state} → {self.drawdown_state}")
            print(f"  └─ Drawdown: {self.daily_drawdown*100:.2f}% | Peak: ${self.peak_balance:.2f} | Current: ${current_balance:.2f}\n")
    
    def should_trade(self) -> bool:
        """Verifica si podemos tradear según el drawdown."""
        if self.drawdown_state == "STOP":
            print("[RISK] ⛔ Trading PAUSADO por exceso de drawdown")
            return False
        return True
    
    def get_risk_limits(self) -> Dict:
        """Retorna límites de riesgo actuales."""
        size_reduction = 0.0
        
        if self.drawdown_state == "ALERT":
            size_reduction = 0.30
        elif self.drawdown_state == "CAUTION":
            size_reduction = 0.50
        elif self.drawdown_state == "DANGER":
            size_reduction = 0.70
        elif self.drawdown_state == "STOP":
            size_reduction = 1.0
        
        return {
            "state": self.drawdown_state,
            "drawdown": self.daily_drawdown,
            "size_reduction": size_reduction,
            "can_trade": self.should_trade(),
            "require_high_confidence": self.drawdown_state in ["CAUTION", "DANGER"]
        }
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas del risk manager."""
        return {
            "trailing_stops_active": len(self.trailing_stops),
            "trailing_stops_hit": self.total_trailing_stops_hit,
            "pyramid_positions": len(self.pyramid_entries),
            "total_pyramids": self.total_pyramids,
            "drawdown_state": self.drawdown_state,
            "daily_drawdown": round(self.daily_drawdown * 100, 2),
            "peak_balance": round(self.peak_balance, 2),
            "current_balance": round(self.current_balance, 2)
        }
    
    def calculate_trade_cost(self, entry_price: float, size: float, exit_price: float = None) -> dict:
        """
        Calcula el costo total de un trade incluyendo fees de entrada y salida.
        
        Args:
            entry_price: Precio de entrada
            size: Tamaño de la posición
            exit_price: Precio de salida (opcional, usa entry_price si no se proporciona)
        
        Returns:
            dict con:
                - entry_fee: Fee de entrada en USD
                - exit_fee: Fee de salida en USD  
                - total_fee: Fee total en USD
                - breakeven_price_long: Precio para breakeven en LONG
                - breakeven_price_short: Precio para breakeven en SHORT
                - min_profit_required: Ganancia mínima para cubrir fees × 2
        """
        if exit_price is None:
            exit_price = entry_price
            
        # Calcular fees (usamos taker_fee porque es más común)
        entry_value = entry_price * size
        entry_fee = entry_value * self.config.taker_fee
        
        exit_value = exit_price * size
        exit_fee = exit_value * self.config.taker_fee
        
        total_fee = entry_fee + exit_fee
        
        # Calcular breakeven prices (precio donde PnL = -fees)
        # Para LONG: necesitamos que (exit - entry) * size > total_fee
        # breakeven = entry + (total_fee / size)
        breakeven_long = entry_price + (total_fee / size)
        
        # Para SHORT: necesitamos que (entry - exit) * size > total_fee
        # breakeven = entry - (total_fee / size)  
        breakeven_short = entry_price - (total_fee / size)
        
        # Ganancia mínima requerida (fees × multiplicador)
        min_profit_required = total_fee * self.config.min_profit_multiplier
        
        return {
            "entry_fee": round(entry_fee, 2),
            "exit_fee": round(exit_fee, 2),
            "total_fee": round(total_fee, 2),
            "breakeven_price_long": round(breakeven_long, 2),
            "breakeven_price_short": round(breakeven_short, 2),
            "min_profit_required": round(min_profit_required, 2)
        }
