"""
Capa de acceso a datos (Supabase / Postgres).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    url: str = "SUPABASE_URL"
    key: str = "SUPABASE_SERVICE_KEY"


def init_db(config: DatabaseConfig | None = None) -> None:
    """Inicializa la conexión a la base de datos."""
    cfg = config or DatabaseConfig()
    # TODO: crear cliente real supabase.Client
    print(f"[db] inicializando conexión a {cfg.url}")


def record_trade(payload: dict) -> None:
    """Guarda un trade ejecutado (demo o real)."""
    print(f"[db] guardando trade: {payload}")


def read_kill_switch() -> bool:
    """Consulta el estado del botón rojo (STOP)."""
    # TODO: leer tabla `settings`
    return False


