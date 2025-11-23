"""
Motor de backtesting para validar estrategias sobre datos históricos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console

from data_collector import get_historical_data, DATA_CACHE
from risk_strategy import RiskStrategy

console = Console()

@dataclass
class BacktestResult:
    final_capital: float
    max_drawdown: float
    trades: int


async def run_backtest() -> BacktestResult:
    """
    Simula la estrategia sobre el dataset histórico disponible.
    """
    # 1. Asegurar datos (intentar descarga, si falla usar cache existente)
    try:
        await get_historical_data()
    except Exception as e:
        console.print(f"[yellow]⚠ No se pudo actualizar datos: {e}. Usando cache local.[/yellow]")

    # 2. Cargar datos de BTC_USD (ejemplo)
    file_path = DATA_CACHE / "BTC_USD_daily.json"
    if not file_path.exists():
        console.print(f"[red]✘ No se encontró {file_path}. Abortando.[/red]")
        return BacktestResult(0, 0, 0)

    data = json.loads(file_path.read_text())
    time_series = data.get("Time Series (Digital Currency Daily)", {})
    
    # Ordenar por fecha ascendente
    sorted_dates = sorted(time_series.keys())
    
    strategy = RiskStrategy()
    capital = strategy.config.capital_virtual
    initial_capital = capital
    peak_capital = capital
    max_dd = 0.0
    trade_count = 0
    position_open = False  # Track if we have an open position
    entry_price = 0.0
    
    console.print(f"[bold]Iniciando simulación con ${capital:,.2f}[/bold]")
    console.print(f"[cyan]Estrategia: SMA Crossover (Short={strategy.short_period}, Long={strategy.long_period})[/cyan]")

    # Alimentar estrategia con datos históricos
    for i, date_str in enumerate(sorted_dates):
        day_data = time_series[date_str]
        close_price = float(day_data.get("4. close", 0))

        
        if close_price <= 0:
            continue
        
        # Obtener señal de la estrategia
        signal = strategy.get_signal(close_price)
        
        # Ejecutar trades basados en señales
        if signal == 'BUY' and not position_open:
            # Abrir posición LONG
            allowed, size, sl, tp = strategy.check_trade("BUY", close_price)
            if allowed:
                entry_price = close_price
                position_open = True
                trade_count += 1
                console.print(f"[green]✓ BUY @ ${close_price:,.2f} on {date_str} (Trade #{trade_count})[/green]")
        
        elif signal == 'SELL' and position_open:
            # Cerrar posición
            pnl = size * (close_price - entry_price)
            capital += pnl
            position_open = False
            
            pnl_pct = ((close_price - entry_price) / entry_price) * 100
            console.print(f"[yellow]✓ SELL @ ${close_price:,.2f} on {date_str} | P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)[/yellow]")
            
            # Update drawdown
            if capital > peak_capital:
                peak_capital = capital
            dd = (peak_capital - capital) / peak_capital
            if dd > max_dd:
                max_dd = dd
    
    # Cerrar posición si queda abierta al final
    if position_open:
        pnl = size * (close_price - entry_price)
        capital += pnl
        console.print(f"[yellow]⚠ Closing final position @ ${close_price:,.2f} | P&L: ${pnl:+,.2f}[/yellow]")
    
    console.print(f"\n[bold]Resultados Finales:[/bold]")
    console.print(f"  Capital Inicial: ${initial_capital:,.2f}")
    console.print(f"  Capital Final: ${capital:,.2f}")
    console.print(f"  Retorno: {((capital - initial_capital) / initial_capital * 100):+.2f}%")
    console.print(f"  Max Drawdown: {max_dd * 100:.2f}%")
    console.print(f"  Total Trades: {trade_count}")

    return BacktestResult(
        final_capital=capital,
        max_drawdown=max_dd,
        trades=trade_count
    )


