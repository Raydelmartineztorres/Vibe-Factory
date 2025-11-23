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
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    Inicia el servidor API (FastAPI + Uvicorn).
    """
    import uvicorn
    console.print(f"[cyan]→ Iniciando servidor API en http://{host}:{port}[/cyan]")
    
    # --- STATIC FILES (Frontend) ---
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    import os
    from api import app # Import the FastAPI app instance

    # Determinar ruta de estáticos (compatible con Docker y local)
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if not os.path.exists(static_dir):
        # Fallback para desarrollo local si se construye en ../frontend/out
        static_dir = os.path.join(os.path.dirname(__file__), "../frontend/out")

    if os.path.exists(static_dir):
        # Mount static assets
        app.mount("/_next", StaticFiles(directory=os.path.join(static_dir, "_next")), name="next")
        
        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            # Si es una API, dejar pasar (ya manejado por las rutas de api.py)
            if full_path.startswith("api/"):
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="No encontrado")
            
            # Root path
            if full_path == "" or full_path == "/":
                return FileResponse(os.path.join(static_dir, "index.html"))
                
            # Servir archivo si existe
            file_path = os.path.join(static_dir, full_path)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                return FileResponse(file_path)
                
            # Si no existe, servir index.html (SPA fallback)
            return FileResponse(os.path.join(static_dir, "index.html"))
    else:
        print("[WARN] No se encontró directorio 'static' o '../frontend/out'. Frontend no servido.")
    
    uvicorn.run("api:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    cli()
