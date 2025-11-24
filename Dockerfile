# Stage 1: Build Frontend
FROM node:20-alpine AS builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Setup Backend & Serve
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Backend Code
COPY backend/ ./backend/

# Copy Built Frontend from Stage 1
COPY --from=builder /app/frontend/out ./backend/static

# Expose Port
EXPOSE 8000

# Run Command
CMD ["python", "backend/main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
