## Vibe Factory – Playbook Operativo

Guía rápida para levantar la fábrica (frontend + backend) y conectar las piezas antes de iniciar un proyecto concreto (por ejemplo, la app de trading).

---

### 1. Preparar variables de entorno

1. Copiar el archivo `env.example` en la raíz y renombrarlo a `.env`.
2. Completar cada clave:
   - `SUPABASE_URL` / `SUPABASE_SERVICE_KEY`
   - `DATA_PROVIDER_API_KEY` (Alpha Vantage, Twelve Data, etc.)
   - `LLM_API_KEY` (Claude, OpenAI…)
   - `BROKER_API_KEY` / `BROKER_API_SECRET`
   - `NEXT_PUBLIC_BACKEND_URL` (URL donde expones el backend)
   - `LIVE_MODE` (empieza siempre en `demo`)

> Consejo: nunca subas `.env` a repositorios públicos.

---

### 2. Frontend (dashboard HITL)

```
cd frontend
npm install
npm run dev
```

Notas:
- Usa `http://localhost:3000` para ver la interfaz base.
- Personaliza `src/app/page.tsx` para añadir componentes (alerts, gráficas, etc.).
- Cuando necesites desplegar, ejecuta `npm run build`.

---

### 3. Backend (módulos IA)

```
cd backend
python3 -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Comandos disponibles:

- `python main.py status` → lista los módulos activados.
- `python main.py backtest` → ejecuta el motor histórico (usa datos fake hasta conectar las APIs reales).
- `python main.py live --mode demo` → inicia el feed en vivo (demo) y enruta ticks a `RiskStrategy`.

Puntos clave:
- Sustituye los TODOs de cada módulo según avances (datos reales, broker, LLM).
- Integra Supabase en `db_interface.py` para persistir settings, trades y el kill switch.

---

### 4. Conectar frontend ↔ backend

- El frontend consume APIs del backend (REST o WebSocket). Configura la URL base en `NEXT_PUBLIC_BACKEND_URL`.
- Desde la UI puedes:
  - Mostrar el estado del bot (riesgo, P&L, señales pendientes).
  - Enviar aprobaciones/overrides (HITL) al backend vía endpoints seguros.

Sugerencia: crea un endpoint `/api/signals` en el backend para que el dashboard lea/escriba decisiones humanas.

---

### 5. Checklist antes de integrar un proyecto nuevo

1. [ ] `.env` completo y verificado.
2. [ ] Frontend corre en local (`npm run dev`) sin errores.
3. [ ] Backend tiene virtualenv y responde a `python main.py status`.
4. [ ] Backtest básico funcionando (aunque sea con datos fake).
5. [ ] Documentar en `docs/` cualquier decisión nueva (por ejemplo, qué API de noticias usas).

Con esta “fábrica” documentada, ya puedes enchufar la idea de trading (u otra app IA) sin rehacer la base cada vez.


