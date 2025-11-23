"""
Sistema de auto-optimización de parámetros.
Ejecuta backtests con diferentes configuraciones y selecciona la mejor.
"""

from __future__ import annotations
from dataclasses import dataclass
import itertools
from typing import List, Dict, Any


@dataclass
class OptimizationResult:
    """Resultado de una optimización de parámetros."""
    params: Dict[str, Any]
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_trades: int
    sharpe_ratio: float
    score: float  # Score compuesto para ranking


class AutoOptimizer:
    """Optimizador automático de parámetros de trading."""
    
    def __init__(self):
        self.best_params = None
        self.optimization_history: List[OptimizationResult] = []
        
    def generate_parameter_grid(self) -> List[Dict[str, Any]]:
        """Genera grid de parámetros para probar."""
        # Definir rangos de parámetros
        rsi_periods = [10, 14, 20]
        atr_periods = [10, 14, 20]
        ema_fast = [5, 9, 12]
        ema_slow = [15, 21, 26]
        macd_fast = [8, 12, 16]
        macd_slow = [20, 26, 32]
        
        # Generar todas las combinaciones (muestreo limitado para eficiencia)
        grid = []
        
        # Tomar muestras en lugar de todas las combinaciones (full grid sería 3^6 = 729)
        # Usamos combinaciones estratégicas
        base_configs = [
            {'rsi_period': 14, 'atr_period': 14, 'ema_fast': 9, 'ema_slow': 21, 'macd_fast': 12, 'macd_slow': 26},  # Config base
            {'rsi_period': 10, 'atr_period': 10, 'ema_fast': 5, 'ema_slow': 15, 'macd_fast': 8, 'macd_slow': 20},   # Rápida
            {'rsi_period': 20, 'atr_period': 20, 'ema_fast': 12, 'ema_slow': 26, 'macd_fast': 16, 'macd_slow': 32},  # Lenta
        ]
        
        # Agregar variaciones de cada config base
        for base in base_configs:
            grid.append(base)
            # Variar RSI
            for rsi in rsi_periods:
                if rsi != base['rsi_period']:
                    variant = base.copy()
                    variant['rsi_period'] = rsi
                    grid.append(variant)
                    
        return grid
        
    def calculate_score(self, 
                       win_rate: float, 
                       profit_factor: float, 
                       max_drawdown: float,
                       total_trades: int,
                       sharpe_ratio: float) -> float:
        """
        Calcula score compuesto para rankear configuraciones.
        
        Ponderación:
        - Win Rate: 25%
        - Profit Factor: 30%
        - Max Drawdown (menor es mejor): 20%
        - Total Trades (suficientes datos): 10%
        - Sharpe Ratio: 15%
        """
        # Normalizar métricas (0-100)
        win_rate_norm = win_rate * 100
        
        # Profit factor: 1.5 = 50, 2.0 = 100, 1.0 = 0
        profit_factor_norm = min((profit_factor - 1.0) * 100, 100)
        
        # Max drawdown: invertido, 5% = 100, 10% = 50, 20% = 0
        max_dd_norm = max(100 - (max_drawdown * 500), 0)
        
        # Total trades: 50+ trades = 100, 10 trades = 20
        trades_norm = min((total_trades / 50) * 100, 100)
        
        # Sharpe ratio: 1.0 = 50, 2.0 = 100, 0.5 = 25
        sharpe_norm = min(sharpe_ratio * 50, 100)
        
        # Score ponderado
        score = (
            win_rate_norm * 0.25 +
            profit_factor_norm * 0.30 +
            max_dd_norm * 0.20 +
            trades_norm * 0.10 +
            sharpe_norm * 0.15
        )
        
        return score
        
    async def optimize(self, backtester, historical_data: List[dict], max_iterations: int = 20) -> Dict[str, Any]:
        """
        Ejecuta optimización de parámetros.
        
        Args:
            backtester: Instancia del backtester
            historical_data: Datos históricos para backtest
            max_iterations: Máximo número de configuraciones a probar
            
        Returns:
            Mejores parámetros encontrados
        """
        from backtester import run_backtest
        
        param_grid = self.generate_parameter_grid()
        
        # Limitar iteraciones
        param_grid = param_grid[:max_iterations]
        
        print(f"[OPTIMIZER] 🔧 Iniciando optimización con {len(param_grid)} configuraciones...")
        
        results = []
        
        for i, params in enumerate(param_grid):
            print(f"[OPTIMIZER] Probando config {i+1}/{len(param_grid)}: {params}")
            
            # Ejecutar backtest con estos parámetros
            try:
                backtest_result = await run_backtest(
                    historical_data, 
                    params=params,
                    quiet=True  # Sin verbose para optimización
                )
                
                # Calcular score
                score = self.calculate_score(
                    backtest_result.get('win_rate', 0),
                    backtest_result.get('profit_factor', 1.0),
                    backtest_result.get('max_drawdown', 1.0),
                    backtest_result.get('total_trades', 0),
                    backtest_result.get('sharpe_ratio', 0)
                )
                
                result = OptimizationResult(
                    params=params,
                    win_rate=backtest_result.get('win_rate', 0),
                    profit_factor=backtest_result.get('profit_factor', 1.0),
                    max_drawdown=backtest_result.get('max_drawdown', 1.0),
                    total_trades=backtest_result.get('total_trades', 0),
                    sharpe_ratio=backtest_result.get('sharpe_ratio', 0),
                    score=score
                )
                
                results.append(result)
                self.optimization_history.append(result)
                
                print(f"  ↳ Score: {score:.2f} | Win Rate: {result.win_rate:.1%} | PF: {result.profit_factor:.2f}")
                
            except Exception as e:
                print(f"  ↳ Error: {e}")
                continue
                
        # Ordenar por score
        results.sort(key=lambda x: x.score, reverse=True)
        
        if results:
            best = results[0]
            self.best_params = best.params
            
            print(f"\n[OPTIMIZER] ✅ Mejor configuración encontrada:")
            print(f"  Parámetros: {best.params}")
            print(f"  Score: {best.score:.2f}")
            print(f"  Win Rate: {best.win_rate:.1%}")
            print(f"  Profit Factor: {best.profit_factor:.2f}")
            print(f"  Max Drawdown: {best.max_drawdown:.1%}")
            print(f"  Trades: {best.total_trades}")
            print(f"  Sharpe: {best.sharpe_ratio:.2f}\n")
            
            return best.params
        else:
            print("[OPTIMIZER] ⚠️ No se encontraron configuraciones válidas")
            return None
            
    def get_top_configs(self, n: int = 5) -> List[OptimizationResult]:
        """Obtiene las top N configuraciones históricas."""
        sorted_history = sorted(self.optimization_history, key=lambda x: x.score, reverse=True)
        return sorted_history[:n]
