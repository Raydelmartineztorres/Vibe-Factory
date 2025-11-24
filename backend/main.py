"""
Punto de entrada del backend Vibe Factory.

Expone una CLI mínima para:
- ejecutar el backtest (`python main.py backtest`);
- iniciar el modo live (`python main.py live`);
- mostrar el estado de los módulos registrados.
"""

from __future__ import annotations

import asyncio
from typing import Literal

import typer
from rich.console import Console

from backtester import run_backtest
from data_collector import bootstrap_data_pipeline
from db_interface import init_db
from risk_strategy import RiskStrategy

cli = typer.Typer(help="CLI de la fábrica backend")
console = Console()


@cli.command()
def status() -> None:
    """Imprime el estado de los módulos registrados."""
    console.rule("[bold]Vibe Factory – Backend")
    console.print(
        "[green]✓[/green] data_collector\n"
        "[green]✓[/green] news_analyzer\n"
        "[green]✓[/green] risk_strategy\n"
        "[yellow]-[/yellow] broker_api_handler (configurar credenciales)\n"
    )


@cli.command()
def backtest() -> None:
    """Ejecuta el flujo de backtesting completo."""
    console.print("[cyan]→ Iniciando backtest histórico...[/cyan]")
    asyncio.run(run_backtest())
    console.print("[green]✔ Backtest completado[/green]")


@cli.command()
def live(
    mode: str = typer.Argument("demo")
) -> None:
    """
    Inicia el modo live (demo o real).

    - demo: solo paper trading usando broker simulado.
    - real: requiere credenciales válidas en .env.
    """

    console.print(f"[cyan]→ Launch mode: {mode}[/cyan]")
    init_db()
    strategy = RiskStrategy()
    
    # Registrar estrategia en API para tracking
    from api import set_strategy_instance
    set_strategy_instance(strategy)
    
    # Iniciar optimizador en segundo plano
    from optimizer import Optimizer
    optimizer = Optimizer(strategy)
    
    # Crear tarea para el optimizador
    loop = asyncio.get_event_loop()
    loop.create_task(optimizer.start_loop())
    
    asyncio.run(bootstrap_data_pipeline(strategy=strategy, live_mode=mode))


@cli.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Inicia el servidor API (FastAPI + Uvicorn).
    """
    import uvicorn
    import os
    console.print(f"[cyan]→ Iniciando servidor API en http://{host}:{port}[/cyan]")
    
    # Disable reload in production for better stability
    reload = os.getenv("RELOAD", "false").lower() == "true"
    uvicorn.run("api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    cli()
