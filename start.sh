#!/bin/bash

# Matar procesos anteriores en estos puertos si existen
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

echo "🚀 Iniciando Vibe Factory..."

# Iniciar Backend en segundo plano
cd backend
source .venv/bin/activate
python main.py serve > ../backend.log 2>&1 &
BACKEND_PID=$!
echo "✅ Backend iniciado (PID: $BACKEND_PID)"

# Iniciar Frontend
cd ../frontend
echo "✅ Iniciando Frontend..."
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!

echo "⏳ Esperando a que cargue..."
sleep 5
open "http://localhost:3000"

# Mantener script corriendo
wait $FRONTEND_PID
