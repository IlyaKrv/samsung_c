FROM python:3.10-slim

WORKDIR /app

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY src/ ./src/
COPY data/ ./data/

# Создаем директории
RUN mkdir -p /app/data/parquet

# HEALTHCHECK (опционально)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Порты
EXPOSE 8501

CMD ["bash", "-c", "python /app/src/process_podcasts.py && streamlit run /app/src/streamlit_app.py --server.port=8501 --server.headless=true --server.address=0.0.0.0"]