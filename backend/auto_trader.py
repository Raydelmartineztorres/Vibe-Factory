"""
Auto-Trading Background Task
Ejecuta la estrategia automáticamente cada 30 segundos cuando el trading está habilitado.
"""
import asyncio
from datetime import datetime

async def auto_trading_loop(strategy_instance, get_trading_enabled_fn):
    """
    Loop continuo que ejecuta la estrategia cada 30 segundos.
    
    Args:
        strategy_instance: Instancia de RiskStrategy
        get_trading_enabled_fn: Función que devuelve True si el trading está habilitado
    """
    print("[AUTO-TRADER] 🤖 Auto-trading loop iniciado")
    
    while True:
        try:
            await asyncio.sleep(30)  # Esperar 30 segundos entre ejecuciones
            
            if not get_trading_enabled_fn():
                # Trading deshabilitado, solo esperar
                continue
            
            # Ejecutar estrategia
            print(f"[AUTO-TRADER] ⏰ {datetime.now().strftime('%H:%M:%S')} - Ejecutando análisis...")
            result = await strategy_instance.execute_strategy()
            
            if result:
                print(f"[AUTO-TRADER] ✅ Trade ejecutado: {result}")
            
        except Exception as e:
            print(f"[AUTO-TRADER] ❌ Error: {e}")
            # Continuar el loop incluso si hay error
            continue
