#!/bin/bash

# Vibe Factory Launcher (Single Window Mode)
# Runs Backend and Frontend in the same terminal window.
# Closing this window stops everything.

PROJECT_DIR="/Users/raydelmartinez/Desktop/cursor/vibe_factory"

# Function to cleanup background processes on exit
cleanup() {
    echo "� Shutting down Vibe Factory..."
    kill $(jobs -p) 2>/dev/null
    exit
}

# Trap interrupt and exit signals
trap cleanup SIGINT SIGTERM EXIT

echo "🚀 Initializing Vibe Factory..."
echo "📂 Project: $PROJECT_DIR"

# 1. Start Backend
echo "🧠 Starting Brain (Backend)..."
cd "$PROJECT_DIR/backend"
# Try to activate venv, silence error if not found (fallback to system python)
source venv/bin/activate 2>/dev/null || true
python3 -m uvicorn api:app --reload --host 0.0.0.0 --port 8000 > "$PROJECT_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo "✅ Backend running (PID: $BACKEND_PID)"

# 2. Start Frontend
echo "🎨 Starting Face (Frontend)..."
cd "$PROJECT_DIR/frontend"
npm run dev > "$PROJECT_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "✅ Frontend running (PID: $FRONTEND_PID)"

# 3. Open Browser
echo "⏳ Waiting for systems to warm up..."
sleep 5
open "http://localhost:3000"

echo "==================================================="
echo "🎉 Vibe Factory is LIVE!"
echo "👉 Dashboard: http://localhost:3000"
echo "📝 Logs are being saved to backend.log and frontend.log"
echo "❌ Close this window to stop all servers."
echo "==================================================="

# Keep script running to maintain background processes
wait
