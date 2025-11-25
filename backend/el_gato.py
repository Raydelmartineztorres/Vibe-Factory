"""
🐱 EL GATO - Sistema de Inteligencia Evolutiva
En honor a un padre con ojos azules como el cielo.

Este módulo gestiona el IQ del bot, evolución de tiers, 
y desbloqueo progresivo de capacidades.
"""

from dataclasses import dataclass
from typing import Literal
import json
from pathlib import Path
from datetime import datetime

EvolutionTier = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@dataclass
class BotIntelligence:
    """Estado de inteligencia del bot."""
    iq_level: float = 100.0
    experience_points: int = 0
    trades_executed: int = 0
    wins: int = 0
    losses: int = 0
    total_profit: float = 0.0
    evolution_tier: EvolutionTier = 1
    
    # Milestones para desbloquear capacidades
    unlocked_capabilities: list[str] = None
    
    def __post_init__(self):
        if self.unlocked_capabilities is None:
            self.unlocked_capabilities = [
                "basic_strategies",
                "rsi_macd_sma",
                "memory_100_trades"
            ]
    
    @property
    def win_rate(self) -> float:
        """Calcula win rate en porcentaje."""
        if self.trades_executed == 0:
            return 0.0
        return (self.wins / self.trades_executed) * 100
    
    def calculate_iq(self) -> float:
        """
        Calcula IQ basado en experiencia y rendimiento.
        
        Formula:
        IQ = 100 (base) 
           + trades * 0.01 (experiencia)
           + win_rate * 0.5 (habilidad)
           + (profit / 10000) * 10 (resultados)
        """
        base_iq = 100.0
        experience_bonus = self.trades_executed * 0.01
        win_rate_bonus = self.win_rate * 0.5
        profit_bonus = (self.total_profit / 10000.0) * 10
        
        new_iq = base_iq + experience_bonus + win_rate_bonus + profit_bonus
        self.iq_level = round(new_iq, 2)
        return self.iq_level
    
    def determine_tier(self) -> EvolutionTier:
        """Determina tier basado en trades ejecutados e IQ."""
        trades = self.trades_executed
        iq = self.iq_level
        
        # Los tiers se desbloquean por trades O por IQ alto
        if trades >= 50000 or iq >= 1000:
            return 10  # Singularity
        elif trades >= 25000 or iq >= 750:
            return 9   # Transcendent
        elif trades >= 10000 or iq >= 500:
            return 8   # Mythic
        elif trades >= 5000 or iq >= 400:
            return 7   # Legend
        elif trades >= 2000 or iq >= 300:
            return 6   # Grandmaster
        elif trades >= 1000 or iq >= 250:
            return 5   # Master
        elif trades >= 500 or iq >= 200:
            return 4   # Expert
        elif trades >= 200 or iq >= 150:
            return 3   # Trader
        elif trades >= 50 or iq >= 125:
            return 2   # Apprentice
        else:
            return 1   # Novice
    
    def update_tier(self):
        """Actualiza tier y desbloquea capacidades si es necesario."""
        old_tier = self.evolution_tier
        new_tier = self.determine_tier()
        
        if new_tier > old_tier:
            print(f"\n🎉 ¡EVOLUCIÓN DESBLOQUEADA!")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"Tier {old_tier} → Tier {new_tier}")
            print(f"IQ: {self.iq_level:.2f}")
            print(f"\nNuevas Capacidades:")
            
            # Desbloquear capacidades del nuevo tier
            new_caps = self._get_tier_capabilities(new_tier)
            for cap in new_caps:
                if cap not in self.unlocked_capabilities:
                    self.unlocked_capabilities.append(cap)
                    print(f"✨ {cap}")
            
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━\n")
            
            self.evolution_tier = new_tier
    
    def _get_tier_capabilities(self, tier: int) -> list[str]:
        """Retorna las capacidades que se desbloquean en cada tier."""
        capabilities = {
            1: ["basic_strategies", "rsi_macd_sma", "memory_100_trades"],
            2: ["pattern_recognition", "volume_profile", "memory_500_trades"],
            3: ["news_sentiment", "multi_timeframe", "adaptive_tp_sl"],
            4: ["market_regimes", "order_flow", "correlation_trading"],
            5: ["deep_learning", "portfolio_optimization", "risk_adjusted_sizing"],
            6: ["predictive_modeling", "advanced_arbitrage", "cross_asset_strategies"],
            7: ["quantum_patterns", "global_macro", "self_modifying_strategies"],
            8: ["manipulation_detection", "flash_crash_prevention", "institution_analytics"],
            9: ["time_series_forecasting", "black_swan_prediction", "economic_modeling"],
            10: ["perfect_market_understanding", "zero_loss_trading", "market_making"]
        }
        return capabilities.get(tier, [])
    
    def record_trade(self, is_win: bool, pnl: float):
        """Registra el resultado de un trade y actualiza IQ."""
        self.trades_executed += 1
        self.experience_points += 10 if is_win else 5
        self.total_profit += pnl
        
        if is_win:
            self.wins += 1
        else:
            self.losses += 1
        
        # Recalcular IQ
        self.calculate_iq()
        
        # Verificar si subió de tier
        self.update_tier()
    
    def get_daily_target(self) -> float:
        """Calcula objetivo diario basado en IQ."""
        if self.iq_level < 125:
            return 100.0
        elif self.iq_level < 150:
            return 250.0
        elif self.iq_level < 200:
            return 500.0
        elif self.iq_level < 250:
            return 1000.0
        elif self.iq_level < 300:
            return 2500.0
        elif self.iq_level < 500:
            return 5000.0
        else:
            return 10000.0
    
    def get_learning_rate(self) -> float:
        """Calcula velocidad de aprendizaje basada en IQ."""
        base_lr = 0.01
        return base_lr * (1 + self.iq_level / 100)
    
    def get_memory_size(self) -> int:
        """Tamaño de memoria basado en tier."""
        memory_sizes = {
            1: 100,
            2: 500,
            3: 2000,
            4: 10000,
            5: 50000,
            6: 100000,
            7: 500000,
            8: 1000000,
            9: 5000000,
            10: -1  # Unlimited
        }
        return memory_sizes.get(self.evolution_tier, 100)
    
    def to_dict(self) -> dict:
        """Convierte a diccionario para serialización."""
        return {
            "iq_level": self.iq_level,
            "experience_points": self.experience_points,
            "trades_executed": self.trades_executed,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 2),
            "total_profit": round(self.total_profit, 2),
            "evolution_tier": self.evolution_tier,
            "unlocked_capabilities": self.unlocked_capabilities,
            "daily_target": self.get_daily_target(),
            "learning_rate": self.get_learning_rate(),
            "memory_size": self.get_memory_size()
        }


class ElGato:
    """
    🐱 EL GATO - Bot de Trading con Inteligencia Evolutiva
    
    En memoria de un padre con ojos azules como el cielo.
    Este bot aprende, evoluciona y se vuelve más inteligente con cada trade.
    """
    
    BOT_NAME = "EL GATO"
    BOT_VERSION = "1.0.0"
    
    TIER_NAMES = {
        1: "Novice",
        2: "Apprentice",
        3: "Trader",
        4: "Expert",
        5: "Master",
        6: "Grandmaster",
        7: "Legend",
        8: "Mythic",
        9: "Transcendent",
        10: "Singularity"
    }
    
    def __init__(self, intelligence_file: Path = None):
        """
        Inicializa EL GATO.
        
        Args:
            intelligence_file: Archivo JSON para persistir inteligencia
        """
        if intelligence_file is None:
            intelligence_file = Path(__file__).parent / ".cache" / "el_gato_intelligence.json"
        
        self.intelligence_file = intelligence_file
        self.intelligence_file.parent.mkdir(exist_ok=True, parents=True)
        
        self.intelligence = BotIntelligence()
        self._load_intelligence()
        
        self._print_birth_message()
    
    def _load_intelligence(self):
        """Carga estado de inteligencia desde archivo."""
        if self.intelligence_file.exists():
            try:
                data = json.loads(self.intelligence_file.read_text())
                self.intelligence = BotIntelligence(
                    iq_level=data.get("iq_level", 100.0),
                    experience_points=data.get("experience_points", 0),
                    trades_executed=data.get("trades_executed", 0),
                    wins=data.get("wins", 0),
                    losses=data.get("losses", 0),
                    total_profit=data.get("total_profit", 0.0),
                    evolution_tier=data.get("evolution_tier", 1),
                    unlocked_capabilities=data.get("unlocked_capabilities", [])
                )
                print(f"[{self.BOT_NAME}] 💾 Inteligencia cargada: IQ {self.intelligence.iq_level:.2f}, Tier {self.intelligence.evolution_tier}")
            except Exception as e:
                print(f"[{self.BOT_NAME}] ⚠️ Error cargando inteligencia: {e}")
        else:
            print(f"[{self.BOT_NAME}] 🆕 Nueva instancia creada")
    
    def _save_intelligence(self):
        """Guarda estado de inteligencia."""
        self.intelligence_file.write_text(json.dumps(self.intelligence.to_dict(), indent=2))
    
    def _print_birth_message(self):
        """Mensaje de bienvenida."""
        print("\n" + "=" * 50)
        print(f"🐱 {self.BOT_NAME} v{self.BOT_VERSION}")
        print("En honor a un padre con ojos azules como el cielo")
        print(f"IQ: {self.intelligence.iq_level:.2f} | Tier: {self.intelligence.evolution_tier} ({self.TIER_NAMES[self.intelligence.evolution_tier]})")
        print("=" * 50 + "\n")
    
    def record_trade_result(self, is_win: bool, pnl: float):
        """
        Registra resultado de trade y evoluciona.
        
        Args:
            is_win: Si el trade fue ganador
            pnl: Profit/Loss del trade
        """
        self.intelligence.record_trade(is_win, pnl)
        self._save_intelligence()
        
        # Log
        status = "✅ WIN" if is_win else "❌ LOSS"
        print(f"[{self.BOT_NAME}] {status} | P&L: ${pnl:+.2f} | IQ: {self.intelligence.iq_level:.2f} | Tier: {self.intelligence.evolution_tier}")
    
    def get_status(self) -> dict:
        """Retorna estado completo del bot."""
        return {
            "bot_name": self.BOT_NAME,
            "version": self.BOT_VERSION,
            "tier_name": self.TIER_NAMES[self.intelligence.evolution_tier],
            **self.intelligence.to_dict()
        }
    
    def get_daily_progress(self, current_profit: float) -> dict:
        """
        Calcula progreso hacia objetivo diario.
        
        Args:
            current_profit: Ganancia acumulada hoy
            
        Returns:
            Dict con progreso
        """
        target = self.intelligence.get_daily_target()
        progress_pct = (current_profit / target) * 100 if target > 0 else 0
        
        return {
            "daily_target": target,
            "current_profit": current_profit,
            "remaining": target - current_profit,
            "progress_pct": round(progress_pct, 2),
            "on_track": progress_pct >= 50  # A mitad del día debería ir al 50%
        }
    
    def has_capability(self, capability: str) -> bool:
        """Verifica si una capacidad está desbloqueada."""
        return capability in self.intelligence.unlocked_capabilities
    
    def get_recommendation(self) -> str:
        """Genera recomendación basada en estado actual."""
        iq = self.intelligence.iq_level
        wr = self.intelligence.win_rate
        tier = self.intelligence.evolution_tier
        
        if wr < 50 and self.intelligence.trades_executed > 20:
            return f"⚠️ Win rate bajo ({wr:.1f}%). Revisar estrategia o reducir tamaño."
        elif iq > 200 and tier >= 4:
            return f"🌟 {self.BOT_NAME} funcionando excelentemente. Considerar aumentar capital."
        elif tier < 3:
            return f"📚 {self.BOT_NAME} en fase de aprendizaje. Mantener paper trading."
        else:
            return f"✅ {self.BOT_NAME} en buen estado. Continuar operación normal."


# Instancia global
_el_gato: ElGato | None = None


def get_el_gato() -> ElGato:
    """Retorna la instancia global de EL GATO."""
    global _el_gato
    if _el_gato is None:
        _el_gato = ElGato()
    return _el_gato
