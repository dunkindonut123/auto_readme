FROM node:20-bookworm-slim AS web-builder

WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOST=0.0.0.0 \
    AUTO_README_HOST=0.0.0.0 \
    AUTO_README_PORT=7860

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py readme_engine.py server.py ./
COPY references ./references
COPY --from=web-builder /app/web/dist ./web/dist

EXPOSE 7860

CMD ["python", "server.py"]