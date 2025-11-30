"""
Módulo de Memoria Avanzado (Reinforcement Learning Mejorado).

Permite al bot recordar resultados de operaciones pasadas bajo ciertas condiciones
de mercado y decidir si evitar operaciones futuras similares.

MEJORAS:
- Memoria expandida con más contexto (RSI, hora, patrones)
- Sistema de confianza basado en cantidad de experiencias
- Decaimiento temporal (experiencias viejas pesan menos)
- Detección de patrones recurrentes
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime, timezone

@dataclass
class MarketContext:
    trend: str      # "UP", "DOWN", "FLAT"
    volatility: str # "HIGH", "LOW", "NORMAL"

@dataclass
class EnhancedContext:
    """Contexto expandido con más información."""
    trend: str
    volatility: str
    rsi_zone: str  # "OVERSOLD" (<30), "NEUTRAL" (30-70), "OVERBOUGHT" (>70)
    hour_utc: int  # Hora del día (0-23)
    pattern: Optional[str] = None  # Patrón de vela detectado
    ml_signal: Optional[str] = None  # "BUY", "SELL", "NEUTRAL"

class TradeMemory:
    def __init__(self):
        self.memory_file = Path(__file__).parent / ".cache" / "trade_memory.json"
        self.experiences = []
        self.pattern_memory: Dict[str, List[float]] = {}  # Patrón -> lista de PnLs
        
        # 🚀 OPTIMIZACIÓN: Cache en RAM para evitar leer archivo constantemente
        self._last_file_load = 0
        self._last_file_save = 0
        self._cache_ttl = 60  # Recargar archivo solo cada 60 segundos
        self._save_delay = 5  # Guardar cambios cada 5 segundos (no en cada operación)
        self._pending_save = False
        
        self._load_memory()

    def _load_memory(self):
        # 🚀 OPTIMIZACIÓN: Solo recargar si el cache expiró
        current_time = time.time()
        if current_time - self._last_file_load < self._cache_ttl:
            return  # Usar datos en RAM
        
        if self.memory_file.exists():
            try:
                data = json.loads(self.memory_file.read_text())
                self.experiences = data.get("experiences", [])
                self.pattern_memory = data.get("patterns", {})
                self._last_file_load = current_time
            except:
                self.experiences = []
                self.pattern_memory = {}

    def _save_memory(self):
        # 🚀 OPTIMIZACIÓN: Guardar con delay para evitar escrituras excesivas
        current_time = time.time()
        
        # Marcar que hay cambios pendientes
        self._pending_save = True
        
        # Solo guardar si pasó el delay o es la primera vez
        if current_time - self._last_file_save < self._save_delay:
            return  # Esperar un poco más
        
        # Guardar solo si hay cambios pendientes
        if not self._pending_save:
            return
        
        self.memory_file.parent.mkdir(exist_ok=True)
        data = {
            "experiences": self.experiences,
            "patterns": self.pattern_memory,
            "last_updated": time.time()
        }
        self.memory_file.write_text(json.dumps(data, indent=2))
        self._last_file_save = current_time
        self._pending_save = False

    def get_context_key(self, context: MarketContext) -> str:
        return f"{context.trend}_{context.volatility}"

    def add_experience(self, context: MarketContext, result_pnl: float, 
                       rsi: float = 50, pattern: str = None, ml_signal: str = None):
        """Registra el resultado de un trade con contexto expandido."""
        
        # Determinar zona RSI
        if rsi < 30:
            rsi_zone = "OVERSOLD"
        elif rsi > 70:
            rsi_zone = "OVERBOUGHT"
        else:
            rsi_zone = "NEUTRAL"
            
        # Hora actual UTC
        hour_utc = datetime.now(timezone.utc).hour
        
        experience = {
            "context": asdict(context),
            "rsi_zone": rsi_zone,
            "rsi_value": round(rsi, 1),
            "hour_utc": hour_utc,
            "pattern": pattern,
            "ml_signal": ml_signal,
            "pnl": result_pnl,
            "outcome": "WIN" if result_pnl > 0 else "LOSS",
            "timestamp": time.time()
        }
        self.experiences.append(experience)
        
        # Guardar en pattern memory si hay patrón
        if pattern:
            if pattern not in self.pattern_memory:
                self.pattern_memory[pattern] = []
            self.pattern_memory[pattern].append(result_pnl)
            # Mantener últimos 100 resultados por patrón (2x aumentado, seguro)
            if len(self.pattern_memory[pattern]) > 100:
                self.pattern_memory[pattern].pop(0)
        
        self._save_memory()
        
        # Log mejorado
        context_str = f"{context.trend}/{context.volatility}/{rsi_zone}"
        pattern_str = f" | Patrón: {pattern}" if pattern else ""
        ml_str = f" | ML: {ml_signal}" if ml_signal else ""
        print(f"[MEMORY] 💾 Experiencia guardada: {context_str}{pattern_str}{ml_str} -> {experience['outcome']} (${result_pnl:.2f})")

    def should_avoid(self, context: MarketContext, rsi: float = 50, 
                     current_hour: int = None, pattern: str = None) -> bool:
        """
        Decide si evitar un trade basado en la historia con contexto expandido.
        Usa decaimiento temporal: experiencias recientes pesan más.
        """
        
        # Determinar zona RSI
        if rsi < 30:
            rsi_zone = "OVERSOLD"
        elif rsi > 70:
            rsi_zone = "OVERBOUGHT"
        else:
            rsi_zone = "NEUTRAL"
            
        if current_hour is None:
            current_hour = datetime.now(timezone.utc).hour
        
        # Filtrar experiencias relevantes con múltiples criterios
        relevant_exps = []
        current_time = time.time()
        
        for e in self.experiences:
            # 2. Check context similarity (if we have context)
            # 2. Check context similarity (if we have context)
            if "context" in e and context:
                e_ctx = e["context"]
                # Safe access to nested keys - String comparison
                if (e_ctx.get("trend") != context.trend or
                    e_ctx.get("volatility") != context.volatility):
                    continue
            elif "context" not in e:
                 # Skip experiences without context if we are looking for specific context
                 continue              
            # Match de zona RSI
            if e.get("rsi_zone") != rsi_zone:
                continue
            
            # Peso por antigüedad (experiencias de más de 7 días pesan menos)
            age_days = (current_time - e.get("timestamp", current_time)) / 86400
            weight = max(1.0 - (age_days / 30), 0.3)  # Mínimo 30% peso
            
            relevant_exps.append((e, weight))

        if not relevant_exps:
            return False  # Sin experiencia, probar suerte

        # 🚀 NUEVO: Calcular PROFIT NETO en lugar de Win Rate
        # Esto prioriza "engordar la cuenta" sobre "ganar más trades"
        total_weight = sum(w for _, w in relevant_exps)
        
        # Calcular profit neto ponderado
        weighted_profit = sum(e.get("pnl", 0) * w for e, w in relevant_exps)
        avg_profit_per_trade = weighted_profit / total_weight if total_weight > 0 else 0
        
        total_experiences = len(relevant_exps)
        
        # Calcular win rate solo para logging (no para decisiones)
        weighted_wins = sum(w for e, w in relevant_exps if "outcome" in e and e["outcome"] == "WIN")
        weighted_win_rate = weighted_wins / total_weight if total_weight > 0 else 0
        
        # Verificar pattern memory si hay patrón
        pattern_suggests_avoid = False
        if pattern and pattern in self.pattern_memory:
            pattern_results = self.pattern_memory[pattern]
            if len(pattern_results) >= 5:
                pattern_avg = sum(pattern_results) / len(pattern_results)
                if pattern_avg < 0:  # Patrón históricamente perdedor
                    pattern_suggests_avoid = True
                    print(f"[MEMORY] ⚠️ Patrón '{pattern}' tiene historial negativo (Avg PnL: ${pattern_avg:.2f})")

        # 🚀 Decision logic NUEVA: Basada en PROFIT NETO, no en Win Rate
        # Objetivo: "Engordar la cuenta", no "ganar más trades"
        confidence_threshold = 10  # 2x: Necesitamos al menos 10 experiencias para alta confianza
        
        if total_experiences >= confidence_threshold:
            # Alta confianza: usar profit neto promedio
            if avg_profit_per_trade < -5:  # Perdiendo $5+ por trade en promedio
                print(f"[MEMORY] 🧠 RECUERDO FUERTE: En {context.trend}/{context.volatility}/{rsi_zone} PIERDO DINERO")
                print(f"  └─ Profit Neto: ${avg_profit_per_trade:.2f}/trade | Win Rate: {weighted_win_rate*100:.1f}% ({total_experiences} exp.)")
                return True
        elif total_experiences >= 3:
            # Confianza media: umbral más permisivo
            if avg_profit_per_trade < -10 or pattern_suggests_avoid:
                print(f"[MEMORY] 🤔 RECUERDO DÉBIL: En {context.trend}/{context.volatility}/{rsi_zone} podría perder")
                print(f"  └─ Profit Neto: ${avg_profit_per_trade:.2f}/trade | Win Rate: {weighted_win_rate*100:.1f}% ({total_experiences} exp.)")
                return True
        
        # Si tenemos buen historial de profit neto, anunciarlo
        if total_experiences >= 3 and avg_profit_per_trade > 10:
            print(f"[MEMORY] ✅ RECUERDO POSITIVO: En {context.trend}/{context.volatility}/{rsi_zone} ENGORDO LA CUENTA")
            print(f"  └─ Profit Neto: +${avg_profit_per_trade:.2f}/trade | Win Rate: {weighted_win_rate*100:.1f}%")
        
        return False

    def get_pattern_stats(self, pattern: str) -> Optional[Dict]:
        """Obtiene estadísticas de un patrón específico."""
        if pattern not in self.pattern_memory or len(self.pattern_memory[pattern]) == 0:
            return None
            
        results = self.pattern_memory[pattern]
        wins = sum(1 for r in results if r > 0)
        avg_pnl = sum(results) / len(results)
        win_rate = (wins / len(results)) * 100
        
        return {
            "pattern": pattern,
            "total_trades": len(results),
            "win_rate": round(win_rate, 1),
            "avg_pnl": round(avg_pnl, 2),
            "confidence": "HIGH" if len(results) >= 10 else "MEDIUM" if len(results) >= 5 else "LOW"
        }

    def get_best_conditions(self) -> Dict:
        """Identifica las mejores condiciones de mercado para tradear."""
        if len(self.experiences) < 20:  # 2x aumentado
            return {}
            
        # Agrupar por condiciones
        conditions = {}
        # Solo procesar experiencias válidas
        valid_exps = [e for e in self.experiences if "outcome" in e and "pnl" in e and "context" in e]
        for exp in valid_exps:
            key = f"{exp['context']['trend']}_{exp['context']['volatility']}_{exp.get('rsi_zone', 'UNKNOWN')}"
            if key not in conditions:
                conditions[key] = {"wins": 0, "total": 0, "pnl": 0}
            conditions[key]["total"] += 1
            if exp["outcome"] == "WIN":
                conditions[key]["wins"] += 1
            conditions[key]["pnl"] += exp["pnl"]
        
        # Calcular métricas
        best_conditions = []
        for key, data in conditions.items():
            if data["total"] >= 10:  # 2x: Solo condiciones con suficiente data
                trend, vol, rsi = key.split("_")
                win_rate = (data["wins"] / data["total"]) * 100
                avg_pnl = data["pnl"] / data["total"]
                
                best_conditions.append({
                    "trend": trend,
                    "volatility": vol,
                    "rsi_zone": rsi,
                    "win_rate": round(win_rate, 1),
                    "avg_pnl": round(avg_pnl, 2),
                    "total_trades": data["total"]
                })
        
        # Ordenar por win rate
        best_conditions.sort(key=lambda x: x["win_rate"], reverse=True)
        
        return {
            "best_3": best_conditions[:3],
            "worst_3": best_conditions[-3:] if len(best_conditions) >= 3 else []
        }

    def get_stats(self):
        """Retorna estadísticas detalladas del aprendizaje."""
        if not self.experiences:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_wins": 0,
                "total_losses": 0,
                "recent_trades": [],
                "best_context": None,
                "worst_context": None,
                "average_pnl": 0,
                "pattern_stats": [],
                "learning_confidence": "NONE"
            }
        
        # Calcular métricas globales
        total = len(self.experiences)
        # Filtrar solo experiencias válidas con campo 'outcome'
        valid_experiences = [e for e in self.experiences if "outcome" in e and "pnl" in e]
        wins = [e for e in valid_experiences if e["outcome"] == "WIN"]
        losses = [e for e in valid_experiences if e["outcome"] == "LOSS"]
        win_rate = (len(wins) / len(valid_experiences)) * 100 if valid_experiences else 0
        
        # Gross Profit/Loss & Profit Factor
        gross_profit = sum(e["pnl"] for e in wins)
        gross_loss = abs(sum(e["pnl"] for e in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
        
        # Cumulative PnL for Chart - Only use valid experiences
        cumulative_pnl = []
        running_pnl = 0
        for i, exp in enumerate(valid_experiences):
            running_pnl += exp["pnl"]
            cumulative_pnl.append({
                "trade": i + 1,
                "pnl": round(running_pnl, 2),
                "outcome": exp["outcome"]
            })

        # Últimos 10 trades válidos
        recent_10 = valid_experiences[-10:] if valid_experiences else []
        
        # PnL promedio
        avg_pnl = sum(e["pnl"] for e in valid_experiences) / len(valid_experiences) if valid_experiences else 0
        
        # Mejor y peor contexto (por win rate) - Solo experiencias válidas
        contexts = {}
        for exp in valid_experiences:
            if "context" not in exp:
                continue
            key = f"{exp['context']['trend']}_{exp['context']['volatility']}"
            if key not in contexts:
                contexts[key] = {"wins": 0, "total": 0, "pnl": 0}
            contexts[key]["total"] += 1
            if exp["outcome"] == "WIN":
                contexts[key]["wins"] += 1
            contexts[key]["pnl"] += exp["pnl"]
        
        # Calcular win rate por contexto
        context_stats = []
        for key, data in contexts.items():
            trend, volatility = key.split("_")
            context_stats.append({
                "context": f"{trend}/{volatility}",
                "win_rate": (data["wins"] / data["total"]) * 100,
                "total_trades": data["total"],
                "total_pnl": data["pnl"]
            })
        
        context_stats.sort(key=lambda x: x["win_rate"], reverse=True)
        
        # Pattern stats
        pattern_stats = []
        for pattern in self.pattern_memory:
            stats = self.get_pattern_stats(pattern)
            if stats:
                pattern_stats.append(stats)
        pattern_stats.sort(key=lambda x: x["win_rate"], reverse=True)
        
        # Learning confidence (2x mejorado)
        if total >= 100:  # 2x de 50
            learning_confidence = "HIGH"
        elif total >= 40:  # 2x de 20
            learning_confidence = "MEDIUM"
        elif total >= 10:  # 2x de 5
            learning_confidence = "LOW"
        else:
            learning_confidence = "MINIMAL"
        
        return {
            "total_trades": total,
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_profit": round(gross_profit - gross_loss, 2),
            "total_wins": len(wins),
            "total_losses": len(losses),
            "cumulative_pnl": cumulative_pnl,
            "recent_trades": [
                {
                    "outcome": e.get("outcome", "UNKNOWN"),
                    "pnl": round(e.get("pnl", 0), 2),
                    "context": f"{e.get('context', {}).get('trend', 'N/A')}/{e.get('context', {}).get('volatility', 'N/A')}"
                } for e in recent_10
            ],
            "best_context": context_stats[0] if context_stats else None,
            "worst_context": context_stats[-1] if context_stats else None,
            "average_pnl": round(avg_pnl, 2),
            "all_contexts": context_stats,
            "pattern_stats": pattern_stats,
            "learning_confidence": learning_confidence,
            "best_conditions": self.get_best_conditions()
        }
