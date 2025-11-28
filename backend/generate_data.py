"""
Script para generar datos históricos rápidamente.
Simula ticks para llenar el gráfico con velas.
"""
import asyncio
import time
from risk_strategy import RiskStrategy

async def generate_historical_data():
    """Genera ~100 velas de datos históricos simulados."""
    
    strategy = RiskStrategy()
    
    # Precio base
    base_price = 87500
    
    print("Generando datos históricos...")
    print("Esto tomará ~30 segundos para generar 100 velas (5 min c/u = ~8 horas de datos)")
    
    current_time = time.time() - (100 * 300)  # Empezar hace 100 periodos de 5min
    
    for i in range(500):  # 500 ticks = ~100 velas de 5s cada una
        # Simular movimiento de precio
        price = base_price + (i * 10) + ((-1) ** i * 50)
        
        payload = {
            "s": "BTC/USDT",
            "p": str(price),
            "E": int(current_time * 1000)
        }
        
        strategy.on_tick(payload)
        current_time += 5  # Avanzar 5 segundos
        
        if i % 50 == 0:
            print(f"Progreso: {i}/500 ticks generados ({len(strategy.candles.get('BTC/USDT', []))} velas)")
        
        await asyncio.sleep(0.01)  # Pequeña pausa para no saturar
    
    print(f"\n✅ Completado! Generadas {len(strategy.candles.get('BTC/USDT', []))} velas")
    print(f"📊 Price history: {len(strategy.price_history.get('BTC/USDT', []))} puntos")

if __name__ == "__main__":
    asyncio.run(generate_historical_data())
