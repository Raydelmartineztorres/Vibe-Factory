## Backend – Vibe Factory Core

Este directorio concentra la lógica de negocio y los módulos IA que alimentan cualquier app construida con la fábrica. Cada archivo representa un componente reutilizable:

| Módulo | Rol | Estado |
| --- | --- | --- |
| `main.py` | Orquestador principal. Arranca servicios, coordina módulos y expone APIs/CLI. | 🚧 |
| `data_collector.py` | Descarga datos históricos (20 años daily) y feed en tiempo real (minuto). | 🚧 |
| `news_analyzer.py` | Llama a modelos LLM para etiquetar noticias globales con un puntaje (-1 a +1). | 🚧 |
| `risk_strategy.py` | Calcula position sizing, stop-loss/take-profit y aplica límites HITL. | 🚧 |
| `backtester.py` | Simula la estrategia sobre datos históricos para validar resiliencia. | 🚧 |
| `broker_api_handler.py` | Adaptadores para enviar órdenes a exchanges/brokers oficiales. | 🚧 |
| `db_interface.py` | Conexión con Supabase (settings, trades, kill switch, logs). | 🚧 |

### Flujo recomendado

1. **Instalar dependencias**: `pip install -r requirements.txt` (ver sección siguiente).
2. **Configurar `.env`**: llaves de APIs de datos, LLMs y broker.
3. **Ejecutar backtest**: `python main.py --run-backtest`.
4. **Lanzar modo live**: `python main.py --mode live` (requiere señales aprobadas desde el frontend HITL).

### Dependencias base

El archivo `requirements.txt` incluye:

- `pandas`, `numpy` – manipulación de datos y cálculos de riesgo.
- `httpx`, `websockets` – streaming y llamadas REST.
- `python-dotenv` – manejo de variables de entorno.
- `supabase`, `sqlalchemy` – persistencia de settings y logs.
- `newsapi` (vía `httpx`) – consumo de titulares globales para sentiment.

> Añade aquí cualquier librería adicional (por ejemplo SDKs de brokers específicos) cuando el proyecto lo necesite.

### Próximos pasos inmediatos

- [ ] Escribir esqueletos iniciales en cada módulo con funciones stub y tipos.
- [ ] Definir CLI básica en `main.py` para `--run-backtest` y `--mode live`.
- [ ] Documentar en `docs/` el flujo de despliegue backend + frontend.
- [ ] Copiar `env.example` a `.env` (raíz del proyecto) y rellenar las claves correspondientes.
- [ ] Configurar `NEWS_API_KEY` para habilitar el módulo de sentiment balanceado.

