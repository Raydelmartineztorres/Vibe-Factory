"""
Módulo de Memoria (Reinforcement Learning Básico).

Permite al bot recordar resultados de operaciones pasadas bajo ciertas condiciones
de mercado y decidir si evitar operaciones futuras similares.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class MarketContext:
    trend: str      # "UP", "DOWN", "FLAT"
    volatility: str # "HIGH", "LOW", "NORMAL"

class TradeMemory:
    def __init__(self):
        self.memory_file = Path(__file__).parent / ".cache" / "trade_memory.json"
        self.experiences = []
        self._load_memory()

    def _load_memory(self):
        if self.memory_file.exists():
            try:
                self.experiences = json.loads(self.memory_file.read_text())
            except:
                self.experiences = []

    def _save_memory(self):
        self.memory_file.parent.mkdir(exist_ok=True)
        self.memory_file.write_text(json.dumps(self.experiences, indent=2))

    def get_context_key(self, context: MarketContext) -> str:
        return f"{context.trend}_{context.volatility}"

    def add_experience(self, context: MarketContext, result_pnl: float):
        """Registra el resultado de un trade."""
        experience = {
            "context": asdict(context),
            "pnl": result_pnl,
            "outcome": "WIN" if result_pnl > 0 else "LOSS"
        }
        self.experiences.append(experience)
        self._save_memory()
        print(f"[MEMORY] Experiencia guardada: {context.trend}/{context.volatility} -> {experience['outcome']} (${result_pnl:.2f})")

    def should_avoid(self, context: MarketContext) -> bool:
        """
        Decide si evitar un trade basado en la historia.
        Retorna True si el win-rate histórico para este contexto es < 40%.
        """
        relevant_exps = [
            e for e in self.experiences 
            if e["context"]["trend"] == context.trend and 
               e["context"]["volatility"] == context.volatility
        ]

        if not relevant_exps:
            return False  # Sin experiencia, probar suerte

        wins = sum(1 for e in relevant_exps if e["outcome"] == "WIN")
        total = len(relevant_exps)
        win_rate = wins / total

        # Si tenemos al menos 3 experiencias y el win rate es malo (< 40%), evitar.
        if total >= 3 and win_rate < 0.4:
            print(f"[MEMORY] 🧠 RECUERDO: En {context.trend}/{context.volatility} suelo perder (Win Rate: {win_rate*100:.1f}%). EVITANDO TRADE.")
            return True
        
        return False

    def get_stats(self):
        return {
            "total_memories": len(self.experiences),
            "recent_outcome": self.experiences[-1]["outcome"] if self.experiences else "N/A"
        }
