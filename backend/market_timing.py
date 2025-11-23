"""
Detector de horas pico y adaptador de parámetros.
Analiza volumen histórico por hora del día para optimizar trading.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict


@dataclass
class MarketSession:
    """Sesión de mercado con características."""
    name: str
    is_peak: bool
    avg_volume: float
    aggressiveness: float  # 0.5 (conservador) a 1.5 (agresivo)


class MarketTiming:
    """Detecta horas pico y ajusta parámetros del bot."""
    
    def __init__(self):
        self.volume_by_hour: Dict[int, list[float]] = defaultdict(list)
        self.max_history_per_hour = 100  # Mantener últimas 100 muestras por hora
        
    def record_volume(self, volume: float, timestamp: float | None = None):
        """Registra volumen por hora del día."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).timestamp()
            
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        hour = dt.hour  # Hora UTC (0-23)
        
        self.volume_by_hour[hour].append(volume)
        
        # Limitar historial
        if len(self.volume_by_hour[hour]) > self.max_history_per_hour:
            self.volume_by_hour[hour].pop(0)
            
    def get_avg_volume_by_hour(self, hour: int) -> float:
        """Obtiene volumen promedio para una hora específica."""
        if hour not in self.volume_by_hour or len(self.volume_by_hour[hour]) == 0:
            return 0.0
        return sum(self.volume_by_hour[hour]) / len(self.volume_by_hour[hour])
        
    def is_peak_hour(self, current_hour: int | None = None) -> bool:
        """Determina si la hora actual es hora pico."""
        if current_hour is None:
            current_hour = datetime.now(timezone.utc).hour
            
        # Horas pico típicas para BTC (UTC):
        # - 13:00-17:00 (Europa/US abierto)
        # - 20:00-00:00 (US trading activo)
        peak_hours = {13, 14, 15, 16, 17, 20, 21, 22, 23, 0}
        
        # Verificar también con volumen histórico
        if len(self.volume_by_hour) >= 12:  # Solo si tenemos suficiente historial
            avg_volume = self.get_avg_volume_by_hour(current_hour)
            overall_avg = sum(
                sum(vols) / len(vols) for vols in self.volume_by_hour.values() if len(vols) > 0
            ) / len(self.volume_by_hour)
            
            # Si volumen actual es 20% mayor que promedio, considerarlo pico
            if avg_volume > overall_avg * 1.2:
                return True
                
        return current_hour in peak_hours
        
    def get_current_session(self) -> MarketSession:
        """Obtiene la sesión de mercado actual con parámetros ajustados."""
        current_hour = datetime.now(timezone.utc).hour
        is_peak = self.is_peak_hour(current_hour)
        avg_vol = self.get_avg_volume_by_hour(current_hour)
        
        if is_peak:
            return MarketSession(
                name="PEAK_HOURS",
                is_peak=True,
                avg_volume=avg_vol,
                aggressiveness=1.3  # Más agresivo en horas pico
            )
        else:
            return MarketSession(
                name="OFF_HOURS",
                is_peak=False,
                avg_volume=avg_vol,
                aggressiveness=0.7  # Más conservador en horas muertas
            )
            
    def adjust_parameters(self, base_params: dict) -> dict:
        """
        Ajusta parámetros de trading según la sesión actual.
        
        Args:
            base_params: dict con 'rsi_threshold', 'volume_threshold', etc.
            
        Returns:
            dict con parámetros ajustados
        """
        session = self.get_current_session()
        adjusted = base_params.copy()
        
        # Ajustar umbrales según agresividad
        if session.is_peak:
            # En horas pico: más permisivo con RSI, menor umbral de volumen
            adjusted['rsi_buy_threshold'] = adjusted.get('rsi_buy_threshold', 70) + 5
            adjusted['rsi_sell_threshold'] = adjusted.get('rsi_sell_threshold', 30) - 5
            adjusted['volume_multiplier'] = adjusted.get('volume_multiplier', 0.5) * 0.8
        else:
            # Fuera de horas pico: más estricto
            adjusted['rsi_buy_threshold'] = adjusted.get('rsi_buy_threshold', 70) - 5
            adjusted['rsi_sell_threshold'] = adjusted.get('rsi_sell_threshold', 30) + 5
            adjusted['volume_multiplier'] = adjusted.get('volume_multiplier', 0.5) * 1.2
            
        return adjusted
        
    def get_stats(self) -> dict:
        """Obtiene estadísticas de volumen por hora."""
        stats = {}
        for hour in range(24):
            avg = self.get_avg_volume_by_hour(hour)
            is_peak = self.is_peak_hour(hour)
            stats[hour] = {
                "avg_volume": avg,
                "is_peak": is_peak,
                "samples": len(self.volume_by_hour.get(hour, []))
            }
        return stats
