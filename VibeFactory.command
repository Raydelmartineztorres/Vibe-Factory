#!/bin/bash

# Vibe Factory Launcher 🚀
# ========================

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}   🚀 VIBE FACTORY LAUNCHER 🚀   ${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# 1. Ir al directorio del proyecto
PROJECT_DIR="/Users/raydelmartinez/Desktop/cursor/vibe_factory"
cd "$PROJECT_DIR" || { echo "❌ Error: No encuentro el directorio del proyecto"; exit 1; }

# 2. Verificar si necesitamos construir el frontend
echo -e "${GREEN}📦 Verificando Frontend...${NC}"
if [ ! -d "frontend/out" ]; then
    echo "   Construyendo Frontend (esto puede tardar un poco la primera vez)..."
    cd frontend
    npm install
    npm run build
    cd ..
else
    echo "   Frontend ya construido. Saltando build."
    # Opcional: Descomentar para forzar rebuild siempre
    # cd frontend && npm run build && cd ..
fi

# 3. Iniciar Backend
echo -e "${GREEN}🧠 Iniciando Cerebro (Backend)...${NC}"
cd backend

# Instalar dependencias si faltan (rápido)
pip install -r requirements.txt > /dev/null 2>&1

# Matar procesos anteriores en puerto 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null

# Iniciar servidor
echo -e "${GREEN}✅ Servidor iniciado en http://localhost:8000${NC}"
echo "   Presiona CTRL+C para detener."
echo ""

# Abrir navegador
open http://localhost:8000

# Ejecutar uvicorn
python3 -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
