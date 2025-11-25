"""
Sistema de perfiles de estrategia para trading bot.
Permite cambiar entre diferentes configuraciones de TP/SL y tracking de resultados.
"""

from dataclasses import dataclass
from typing import Literal
import json
from pathlib import Path
from datetime import datetime

StrategyName = Literal["AGGRESSIVE", "BALANCED", "CONSERVATIVE"]


@dataclass
class StrategyProfile:
    """Configuración de una estrategia de trading."""
    name: StrategyName
    take_profit_pct: float  # Porcentaje de ganancia objetivo
    stop_loss_pct: float    # Porcentaje de pérdida máxima
    min_confidence: float   # Confianza mínima ML para entrar (0-1)
    description: str
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calcula el ratio riesgo/recompensa."""
        return abs(self.take_profit_pct / self.stop_loss_pct)


@dataclass
class StrategyStats:
    """Estadísticas de rendimiento de una estrategia."""
    strategy_name: StrategyName
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    peak_capital: float = 10000.0  # Capital inicial
    is_active: bool = False
    is_paused: bool = False
    
    @property
    def win_rate(self) -> float:
        """Retorna el win rate (0-100)."""
        if self.total_trades == 0:
            return 0.0
        return (self.wins / self.total_trades) * 100
    
    @property
    def avg_profit_per_trade(self) -> float:
        """Retorna el promedio de ganancia por trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades
    
    @property
    def roi(self) -> float:
        """Retorna el ROI (Return on Investment) en porcentaje."""
        return (self.total_pnl / 10000.0) * 100
    
    def record_trade(self, pnl: float, current_capital: float):
        """Registra el resultado de un trade."""
        self.total_trades += 1
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        
        self.total_pnl += pnl
        
        # Actualizar drawdown
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = ((self.peak_capital - current_capital) / self.peak_capital) * 100
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
    
    def to_dict(self) -> dict:
        """Convierte stats a diccionario."""
        return {
            "strategy_name": self.strategy_name,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 2),
            "total_pnl": round(self.total_pnl, 2),
            "avg_profit_per_trade": round(self.avg_profit_per_trade, 2),
            "roi": round(self.roi, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "current_drawdown": round(self.current_drawdown, 2),
            "is_active": self.is_active,
            "is_paused": self.is_paused
        }


class StrategyManager:
    """Maneja las estrategias disponibles y sus estadísticas."""
    
    # Definición de las 3 estrategias
    STRATEGIES = {
        "AGGRESSIVE": StrategyProfile(
            name="AGGRESSIVE",
            take_profit_pct=0.4,  # Optimized: +0.4% (was 0.5%)
            stop_loss_pct=0.2,    # Optimized: -0.2% (was 0.3%)
            min_confidence=0.6,
            description="Scalping rápido: muchos trades pequeños"
        ),
        "BALANCED": StrategyProfile(
            name="BALANCED",
            take_profit_pct=1.5,  # Optimized: +1.5% (was 2.0%)
            stop_loss_pct=0.7,    # Optimized: -0.7% (was 1.0%)
            min_confidence=0.7,
            description="Swing trading: balance riesgo/recompensa"
        ),
        "CONSERVATIVE": StrategyProfile(
            name="CONSERVATIVE",
            take_profit_pct=4.0,  # Optimized: +4.0% (was 5.0%)
            stop_loss_pct=1.8,    # Optimized: -1.8% (was 2.0%)
            min_confidence=0.8,
            description="Position trading: pocos trades grandes"
        )
    }
    
    def __init__(self, stats_file: Path = None):
        """
        Inicializa el gestor de estrategias.
        
        Args:
            stats_file: Archivo JSON para guardar estadísticas
        """
        if stats_file is None:
            stats_file = Path(__file__).parent / ".cache" / "strategy_stats.json"
        
        self.stats_file = stats_file
        self.stats_file.parent.mkdir(exist_ok=True, parents=True)
        
        # Stats por estrategia
        self.stats: dict[StrategyName, StrategyStats] = {}
        self._load_stats()
        
        # Estrategia activa (por defecto: BALANCED)
        self.active_strategy: StrategyName = "BALANCED"
        self._ensure_one_active()
    
    def _load_stats(self):
        """Carga estadísticas desde archivo."""
        if self.stats_file.exists():
            try:
                data = json.loads(self.stats_file.read_text())
                for name in ["AGGRESSIVE", "BALANCED", "CONSERVATIVE"]:
                    if name in data:
                        stats_dict = data[name]
                        self.stats[name] = StrategyStats(
                            strategy_name=name,
                            total_trades=stats_dict.get("total_trades", 0),
                            wins=stats_dict.get("wins", 0),
                            losses=stats_dict.get("losses", 0),
                            total_pnl=stats_dict.get("total_pnl", 0.0),
                            max_drawdown=stats_dict.get("max_drawdown", 0.0),
                            current_drawdown=stats_dict.get("current_drawdown", 0.0),
                            peak_capital=stats_dict.get("peak_capital", 10000.0),
                            is_active=stats_dict.get("is_active", False),
                            is_paused=stats_dict.get("is_paused", False)
                        )
                    else:
                        self.stats[name] = StrategyStats(strategy_name=name)
            except Exception as e:
                print(f"[STRATEGY] Error loading stats: {e}")
                self._initialize_default_stats()
        else:
            self._initialize_default_stats()
    
    def _initialize_default_stats(self):
        """Inicializa stats por defecto."""
        for name in ["AGGRESSIVE", "BALANCED", "CONSERVATIVE"]:
            self.stats[name] = StrategyStats(strategy_name=name)
    
    def _save_stats(self):
        """Guarda estadísticas a archivo."""
        data = {name: stats.to_dict() for name, stats in self.stats.items()}
        self.stats_file.write_text(json.dumps(data, indent=2))
    
    def _ensure_one_active(self):
        """Asegura que exactamente una estrategia esté activa."""
        # Marcar la activa
        for name in self.stats:
            self.stats[name].is_active = (name == self.active_strategy)
        self._save_stats()
    
    def get_active_strategy(self) -> StrategyProfile:
        """Retorna el perfil de la estrategia activa."""
        return self.STRATEGIES[self.active_strategy]
    
    def get_active_stats(self) -> StrategyStats:
        """Retorna las stats de la estrategia activa."""
        return self.stats[self.active_strategy]
    
    def switch_strategy(self, name: StrategyName) -> bool:
        """
        Cambia la estrategia activa.
        
        Args:
            name: Nombre de la estrategia
            
        Returns:
            True si se cambió exitosamente
        """
        if name not in self.STRATEGIES:
            return False
        
        if self.stats[name].is_paused:
            print(f"[STRATEGY] ⚠️ {name} está pausada. Despausa primero.")
            return False
        
        old = self.active_strategy
        self.active_strategy = name
        self._ensure_one_active()
        
        print(f"[STRATEGY] 🔄 Cambiado de {old} → {name}")
        return True
    
    def pause_strategy(self, name: StrategyName):
        """Pausa una estrategia (no se puede activar hasta despausar)."""
        if name in self.stats:
            self.stats[name].is_paused = True
            if name == self.active_strategy:
                # Si pausamos la activa, cambiar a BALANCED
                self.active_strategy = "BALANCED"
                self._ensure_one_active()
            self._save_stats()
            print(f"[STRATEGY] ⏸️ {name} pausada")
    
    def resume_strategy(self, name: StrategyName):
        """Despausa una estrategia."""
        if name in self.stats:
            self.stats[name].is_paused = False
            self._save_stats()
            print(f"[STRATEGY] ▶️ {name} reanudada")
    
    def record_trade_result(self, pnl: float, current_capital: float):
        """
        Registra el resultado de un trade en la estrategia activa.
        
        Args:
            pnl: Profit/Loss del trade
            current_capital: Capital actual después del trade
        """
        active_stats = self.get_active_stats()
        active_stats.record_trade(pnl, current_capital)
        self._save_stats()
        
        # Log resultado
        status = "✅ WIN" if pnl > 0 else "❌ LOSS"
        print(f"[STRATEGY {self.active_strategy}] {status} | P&L: ${pnl:+.2f} | Total: ${active_stats.total_pnl:+.2f} | WR: {active_stats.win_rate:.1f}%")
    
    def get_all_stats(self) -> dict:
        """Retorna stats de todas las estrategias."""
        return {
            "strategies": [stats.to_dict() for stats in self.stats.values()],
            "active_strategy": self.active_strategy
        }
    
    def get_recommendation(self) -> str:
        """Genera recomendación basada en performance."""
        sorted_strategies = sorted(
            self.stats.values(),
            key=lambda s: (s.win_rate, s.total_pnl),
            reverse=True
        )
        
        best = sorted_strategies[0]
        
        if best.total_trades < 10:
            return "⏳ Necesitas más trades para una recomendación confiable (mínimo 10)"
        
        if best.win_rate >= 60 and best.total_pnl > 0:
            return f"⭐ {best.strategy_name} está funcionando bien ({best.win_rate:.1f}% WR, ${best.total_pnl:+.2f})"
        elif best.win_rate >= 50 and best.total_pnl > 0:
            return f"👍 {best.strategy_name} es decente ({best.win_rate:.1f}% WR, ${best.total_pnl:+.2f})"
        else:
            return f"⚠️ Todas las estrategias perdiendo. Mejor pausar auto-trading y revisar parámetros."


# Instancia global del gestor
_strategy_manager: StrategyManager | None = None


def get_strategy_manager() -> StrategyManager:
    """Retorna la instancia global del gestor."""
    global _strategy_manager
    if _strategy_manager is None:
        _strategy_manager = StrategyManager()
    return _strategy_manager
