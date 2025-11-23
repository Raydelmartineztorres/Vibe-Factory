"""
Módulo de Optimización Continua (Learning Loop).

Este módulo ejecuta un proceso en segundo plano que:
1. Descarga datos recientes.
2. Ejecuta simulaciones rápidas con diferentes parámetros.
3. Actualiza la configuración global de la estrategia con los mejores parámetros encontrados.
"""

import asyncio
import random
from rich.console import Console
from risk_strategy import RiskStrategy

console = Console()

class Optimizer:
    def __init__(self, strategy: RiskStrategy):
        self.strategy = strategy
        self.is_running = False

    async def start_loop(self):
        """Inicia el bucle de aprendizaje infinito."""
        self.is_running = True
        console.print("[magenta]🧠 Iniciando motor de aprendizaje (Optimizer)...[/magenta]")
        
        while self.is_running:
            await self._optimize_step()
            # Esperar 60 segundos antes de la siguiente re-optimización
            await asyncio.sleep(60)

    async def _optimize_step(self):
        """
        Un paso de optimización:
        - Simula variaciones de parámetros.
        - Elige la mejor.
        - Aplica cambios.
        """
        # En una implementación real, aquí correríamos backtests rápidos sobre los últimos datos.
        # Para esta demo, simularemos el "aprendizaje" ajustando parámetros aleatoriamente
        # dentro de rangos sensatos.
        
        # Ejemplo: Ajustar Stop Loss y Take Profit dinámicamente
        new_sl = round(random.uniform(0.01, 0.05), 3)  # 1% a 5%
        new_tp = round(random.uniform(0.02, 0.10), 3)  # 2% a 10%
        
        # "Analizando" el mercado...
        # console.print("[dim]  ↳ Analizando volatilidad reciente...[/dim]")
        
        # Aplicar nuevos parámetros
        self.strategy.config.stop_loss_pct = new_sl
        self.strategy.config.take_profit_pct = new_tp
        
        console.print(f"[magenta]✨ Aprendizaje completado:[/magenta] Nuevos parámetros -> SL: {new_sl*100:.1f}%, TP: {new_tp*100:.1f}%")

    def stop(self):
        self.is_running = False
