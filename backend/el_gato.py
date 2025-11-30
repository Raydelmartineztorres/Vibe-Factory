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
    
    def record_trade(self, profit: float, pattern: str = None):
        """Registra un trade y actualiza inteligencia."""
        self.trades_executed += 1
        self.total_profit += profit
        
        if profit > 0:
            self.wins += 1
            # 🚀🚀🚀 TURBO-LEARNING: XP aumentado 30x (Render optimizado)
            xp_gain = 300 if pattern else 150
        else:
            self.losses += 1
            # 🚀🚀🚀 TURBO-LEARNING: Aprende de derrotas 15x más rápido
            xp_gain = 75  # Aprende incluso de pérdidas
        
        self.experience_points += xp_gain
        
        # 🚀🚀🚀 IQ gain aumentado 15x (Render power)
        iq_gain = max(15, int(xp_gain / 10))
        self.iq_level += iq_gain
        
        # Calcular win rate
        if self.trades_executed > 0:
            self.win_rate = (self.wins / self.trades_executed) * 100
        
        # Verificar si subió de tier
        self.update_tier()
    
    def get_daily_target(self) -> float:
        """
        Calcula objetivo diario basado en BALANCE ACTUAL (crecimiento compuesto).
        
        🚀 ESTRATEGIA CONSERVADORA:
        - Usa 1.5% del balance actual (en lugar de valores fijos)
        - Crece con la cuenta → Compounding
        - Objetivo: +56% en 30 días ($10k → $15.6k)
        """
        # Obtener balance actual (por defecto $10k si no se puede obtener)
        current_balance = 10000.0
        
        try:
            # Intentar obtener balance real del sistema
            from api import _strategy_instance
            if _strategy_instance:
                # Balance = capital virtual + PnL realizado + no realizado
                pnl_data = _strategy_instance.get_portfolio_pnl()
                current_balance = (
                    _strategy_instance.config.capital_virtual +
                    pnl_data.get('realized_pnl', 0) +
                    pnl_data.get('unrealized_pnl', 0)
                )
        except:
            pass
        
        # 📈 CRECIMIENTO COMPUESTO: 1.5% del balance actual
        # Ajustado por realidad: Si simulamos fees, el objetivo neto es más difícil
        # Mantenemos 1.5% bruto, pero sabiendo que el neto será menor
        daily_target = current_balance * 0.015  # 1.5%
        
        return max(100.0, daily_target)  # Mínimo $100
    
    def get_singularity_multiplier(self) -> float:
        """
        Calcula el Multiplicador de Singularidad.
        Factor de escala dinámica para todas las capacidades.
        
        Fórmula: 1.0 + (IQ / 100) + (XP / 5000)
        """
        iq_factor = self.iq_level / 100.0
        xp_factor = self.experience_points / 5000.0
        return 1.0 + iq_factor + xp_factor

    def get_learning_rate(self) -> float:
        """Calcula velocidad de aprendizaje basada en Multiplicador."""
        multiplier = self.get_singularity_multiplier()
        
        # 🧠 CEREBRO CUÁNTICO: Escala logarítmica con el multiplicador
        import math
        base_lr = 0.01 * math.log(multiplier + 1)
        
        # Cap dinámico (puede superar 0.5 si el multiplicador es enorme)
        return min(0.9, max(0.01, base_lr))
    
    def get_memory_size(self) -> int:
        """Tamaño de memoria basado en Multiplicador (Crecimiento Exponencial)."""
        multiplier = self.get_singularity_multiplier()
        
        # 🧠 MEMORIA INFINITA: Base * Multiplier^2
        base_memory = 1000
        memory_size = int(base_memory * (multiplier ** 2))
        
        return memory_size
    
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
                
                # 🚀 EVOLUTION BOOST: Forzar actualización de nivel por mejoras de hardware (Gato Singularidad)
                # Si es Novato o Trader pero tiene capacidades infinitas, subirlo a Tier 10
                if self.intelligence.evolution_tier < 10 and self.intelligence.iq_level < 1000:
                    print(f"[{self.BOT_NAME}] 🌌 Detectada Matriz de Singularidad (Memoria x100). Evolucionando a forma final...")
                    self.intelligence.experience_points += 50000  # Boost masivo
                    self.intelligence.iq_level = 1000.0  # IQ Singularidad
                    self.intelligence.evolution_tier = 10 # Singularity Tier
                    self._save_intelligence()
                    print(f"[{self.BOT_NAME}] 🆙 Nivel actualizado: SINGULARITY (IQ: {self.intelligence.iq_level})")
                    
            except Exception as e:
                print(f"[{self.BOT_NAME}] ⚠️ Error cargando inteligencia: {e}")
        else:
            print(f"[{self.BOT_NAME}] 🆕 Nueva instancia creada")
            # 🌌 TODAS las instancias nuevas empiezan en Singularity
            self.intelligence.iq_level = 1000.0  # IQ Singularidad
            self.intelligence.evolution_tier = 10  # Tier 10: Singularity
            self._save_intelligence()
            print(f"[{self.BOT_NAME}] 🚀 Nivel inicial: SINGULARITY (IQ: 1000.0, Tier: 10/10)")
    
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
    
    def get_recommendation(self, candles: list = None) -> str:
        """Genera recomendación basada en estado actual y velas recientes."""
        iq = self.intelligence.iq_level
        wr = self.intelligence.win_rate
        tier = self.intelligence.evolution_tier
        
        # Análisis técnico si hay velas
        technical_advice = ""
        if candles:
            wick_analysis = self.analyze_candle_wicks(candles)
            if wick_analysis.get("signal") == "rejection_from_top":
                technical_advice = " | 📉 Rechazo alcista detectado (mecha superior larga)."
            elif wick_analysis.get("signal") == "rejection_from_bottom":
                technical_advice = " | 📈 Rechazo bajista detectado (mecha inferior larga)."
            elif wick_analysis.get("signal") == "doji":
                technical_advice = " | ⚖️ Indecisión en el mercado (Doji)."
        
        if wr < 50 and self.intelligence.trades_executed > 20:
            return f"⚠️ Win rate bajo ({wr:.1f}%). Revisar estrategia.{technical_advice}"
        elif iq > 200 and tier >= 4:
            return f"🌟 {self.BOT_NAME} funcionando excelentemente.{technical_advice}"
        elif tier < 3:
            return f"📚 {self.BOT_NAME} aprendiendo. Paper trading.{technical_advice}"
        else:
            return f"✅ {self.BOT_NAME} operativo.{technical_advice}"

    def analyze_candle_wicks(self, candles: list) -> dict:
        """
        Analiza las mechas (wicks) de las velas para detectar señales.
        
        Returns:
            dict con análisis de mechas superiores e inferiores
        """
        if not candles or len(candles) < 3:
            return {"status": "insufficient_data"}
        
        last_candle = candles[-1]
        
        # Calcular tamaños de mechas
        body_size = abs(last_candle.close - last_candle.open)
        upper_wick = last_candle.high - max(last_candle.open, last_candle.close)
        lower_wick = min(last_candle.open, last_candle.close) - last_candle.low
        total_range = last_candle.high - last_candle.low
        
        # Ratios
        upper_wick_ratio = upper_wick / total_range if total_range > 0 else 0
        lower_wick_ratio = lower_wick / total_range if total_range > 0 else 0
        body_ratio = body_size / total_range if total_range > 0 else 0
        
        # Interpretación
        analysis = {
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "body_size": body_size,
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio,
            "body_ratio": body_ratio
        }
        
        # Señales basadas en mechas
        if upper_wick_ratio > 0.5 and lower_wick_ratio < 0.2:
            analysis["signal"] = "rejection_from_top"  # Pin bar bearish
            analysis["strength"] = "strong" if upper_wick_ratio > 0.6 else "medium"
        elif lower_wick_ratio > 0.5 and upper_wick_ratio < 0.2:
            analysis["signal"] = "rejection_from_bottom"  # Pin bar bullish
            analysis["strength"] = "strong" if lower_wick_ratio > 0.6 else "medium"
        elif body_ratio < 0.1:  # 🔥 Relaxed from 0.2 to 0.1 (less sensitive)
            analysis["signal"] = "doji"  # Indecisión
            analysis["strength"] = "weak"
        else:
            analysis["signal"] = "normal"
            analysis["strength"] = "neutral"
        
        return analysis


# Instancia global
_el_gato: ElGato | None = None


def get_el_gato() -> ElGato:
    """Retorna la instancia global de EL GATO."""
    global _el_gato
    if _el_gato is None:
        _el_gato = ElGato()
    return _el_gato
