# Vibe Factory

Plataforma base para crear aplicaciones de trading y análisis con IA.

## Estructura

-   **frontend/**: Next.js + Tailwind CSS. Dashboard para control humano (HITL).
-   **backend/**: Python + FastAPI. Módulos de datos, riesgo y ejecución.

## Cómo iniciar

### Prerrequisitos

-   Node.js (v18+)
-   Python (v3.10+)

### 1. Configuración inicial

Si es la primera vez que descargas el proyecto:

```bash
# 1. Configurar Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env  # Y edita las claves si es necesario

# 2. Configurar Frontend
cd ../frontend
npm install
```

### 2. Ejecutar la aplicación

Necesitas dos terminales abiertas:

**Terminal 1: Backend**

```bash
cd backend
source .venv/bin/activate
python main.py serve
```

_El backend correrá en http://127.0.0.1:8000_

**Terminal 2: Frontend**

```bash
cd frontend
npm run dev
```

_El frontend correrá en http://localhost:3000_

## Comandos útiles del Backend

-   `python main.py status`: Ver estado de módulos.
-   `python main.py backtest`: Ejecutar simulación histórica.
-   `python main.py live`: Iniciar modo en vivo (demo).
