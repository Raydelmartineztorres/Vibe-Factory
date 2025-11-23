"""
Sistema de Entrenamiento Rápido (Fast Training).

Permite al bot aprender de datos históricos ejecutando backtests automáticos
y poblando la memoria con experiencia real en minutos.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict
from pathlib import Path
import sys

# Importar módulos del bot
from data_collector import DataCollector
from risk_strategy import RiskStrategy
from memory import TradeMemory, MarketContext
from broker_api_handler import execute_order

class FastTrainer:
    """Entrena el bot rápidamente usando datos históricos."""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.memory = TradeMemory()
        self.stats = {
            "trades_executed": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "start_time": None,
            "end_time": None
        }
    
    async def run_quick_training(self, num_trades: int = 100, symbol: str = "BTC_USD"):
        """
        Ejecuta entrenamiento rápido simulando trades históricos.
        
        Args:
            num_trades: Número aproximado de trades a simular
            symbol: Par de trading (default: BTC_USD)
        """
        print(f"\n{'='*60}")
        print(f"🚀 ENTRENAMIENTO RÁPIDO INICIADO")
        print(f"{'='*60}")
        print(f"Objetivo: ~{num_trades} trades")
        print(f"Symbol: {symbol}\n")
        
        self.stats["start_time"] = time.time()
        
        # 1. Descargar datos históricos
        print("📊 Descargando datos históricos...")
        days_back = min(180, num_trades // 2)  # ~2 trades por día
        candles = await self._fetch_historical_data(symbol, days=days_back)
        print(f"✅ Descargados {len(candles)} candles ({days_back} días)\n")
        
        # 2. Inicializar estrategia
        print("🧠 Inicializando estrategia de trading...")
        strategy = RiskStrategy(
            initial_capital=10000.0,
            risk_per_trade=0.02,
            ml_enabled=True,
            memory_enabled=True
        )
        print("✅ Estrategia lista\n")
        
        # 3. Ejecutar backtest
        print("⚡ Ejecutando backtest en modo rápido...\n")
        await self._execute_backtest(strategy, candles, num_trades)
        
        # 4. Reportar resultados
        self.stats["end_time"] = time.time()
        self._print_results()
        
        return self.stats
    
    async def _fetch_historical_data(self, symbol: str, days: int) -> List[Dict]:
        """Descarga datos históricos."""
        try:
            # Usar el data collector existente
            candles = await self.data_collector.fetch_candles(
                symbol=symbol,
                interval="1h",  # Velas de 1 hora
                limit=days * 24
            )
            return candles
        except Exception as e:
            print(f"❌ Error descargando datos: {e}")
            print("📌 Usando datos de cache si están disponibles...")
            # Fallback: intentar cargar desde cache
            cache_path = Path(__file__).parent / ".cache" / f"{symbol}_daily.json"
            if cache_path.exists():
                import json
                return json.loads(cache_path.read_text())
            else:
                raise Exception("No hay datos históricos disponibles")
    
    async def _execute_backtest(self, strategy: RiskStrategy, candles: List[Dict], target_trades: int):
        """Ejecuta el backtest con los candles históricos."""
        
        total_candles = len(candles)
        trades_executed = 0
        progress_interval = max(10, target_trades // 10)  # Reportar cada 10%
        
        for i, candle in enumerate(candles):
            # Simular tick
            await strategy.on_tick(candle)
            
            # Contar trades ejecutados
            current_trades = len(strategy.trades)
            if current_trades > trades_executed:
                trades_executed = current_trades
                self.stats["trades_executed"] = trades_executed
                
                # Calcular estadísticas
                last_trade = strategy.trades[-1]
                if last_trade.get("pnl", 0) > 0:
                    self.stats["wins"] += 1
                else:
                    self.stats["losses"] += 1
                
                self.stats["total_pnl"] = strategy.realized_pnl
                
                # Progreso
                if trades_executed % progress_interval == 0:
                    elapsed = time.time() - self.stats["start_time"]
                    win_rate = (self.stats["wins"] / trades_executed * 100) if trades_executed > 0 else 0
                    print(f"  📈 Progreso: {trades_executed}/{target_trades} trades | "
                          f"Win Rate: {win_rate:.1f}% | "
                          f"PnL: ${self.stats['total_pnl']:.2f} | "
                          f"Tiempo: {elapsed:.1f}s")
            
            # Parar si llegamos al objetivo
            if trades_executed >= target_trades:
                print(f"\n✅ Objetivo alcanzado: {trades_executed} trades ejecutados")
                break
        
        # Guardar memoria final
        print(f"\n💾 Guardando experiencias en memoria...")
        # La memoria ya se guarda automáticamente en cada trade
    
    def _print_results(self):
        """Imprime resumen de resultados."""
        elapsed = self.stats["end_time"] - self.stats["start_time"]
        win_rate = (self.stats["wins"] / self.stats["trades_executed"] * 100) if self.stats["trades_executed"] > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"✨ ENTRENAMIENTO COMPLETADO")
        print(f"{'='*60}")
        print(f"⏱️  Tiempo total:        {elapsed:.1f} segundos")
        print(f"📊 Trades ejecutados:   {self.stats['trades_executed']}")
        print(f"✅ Wins:                {self.stats['wins']}")
        print(f"❌ Losses:              {self.stats['losses']}")
        print(f"📈 Win Rate:            {win_rate:.1f}%")
        print(f"💰 PnL Total:           ${self.stats['total_pnl']:.2f}")
        print(f"⚡ Velocidad:           {self.stats['trades_executed'] / elapsed:.1f} trades/seg")
        print(f"{'='*60}\n")
        
        # Verificar que la memoria tenga datos
        memory_file = Path(__file__).parent / ".cache" / "trade_memory.json"
        if memory_file.exists():
            import json
            data = json.loads(memory_file.read_text())
            experiences = len(data.get("experiences", []))
            print(f"💾 Experiencias guardadas: {experiences}")
            print(f"🧠 El bot ahora tiene memoria histórica y puede tomar mejores decisiones!\n")


async def main():
    """Ejecuta entrenamiento rápido standalone."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenamiento rápido del bot")
    parser.add_argument("--trades", type=int, default=100, help="Número de trades objetivo")
    parser.add_argument("--symbol", type=str, default="BTC_USD", help="Par de trading")
    
    args = parser.parse_args()
    
    trainer = FastTrainer()
    await trainer.run_quick_training(num_trades=args.trades, symbol=args.symbol)


if __name__ == "__main__":
    asyncio.run(main())
