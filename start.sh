#!/bin/bash

echo "🧹 Limpiando procesos anteriores..."

# Matar TODOS los procesos de python y node relacionados con el proyecto
pkill -9 -f "python.*main.py" 2>/dev/null
pkill -9 -f "node.*next" 2>/dev/null
pkill -9 -f "uvicorn" 2>/dev/null

# Por si acaso, matar por puertos también
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

# Esperar a que se liberen los recursos
sleep 2

# Limpiar cache de Python
find backend -name "__pycache__" -exec rm -rf {} + 2>/dev/null

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

echo "✅ Todo listo en http://localhost:3000"
echo "📊 Backend: PID $BACKEND_PID (http://localhost:8000)"
echo "🎨 Frontend: PID $FRONTEND_PID (http://localhost:3000)"
echo ""
echo "Para detener: pkill -9 python3; pkill -9 node"

# Mantener script corriendo
wait $FRONTEND_PID
