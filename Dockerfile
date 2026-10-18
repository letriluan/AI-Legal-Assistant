FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_MODE=preview PREVIEW_DB_PATH=/app/preview.sqlite
COPY pyproject.toml requirements-lock.txt ./
COPY src/ src/
COPY archive/ archive/
COPY scripts/build_preview_db.py scripts/start_server.py scripts/
RUN pip install --no-cache-dir -c requirements-lock.txt . && python scripts/build_preview_db.py && chmod 444 /app/preview.sqlite && useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["python", "scripts/start_server.py"]
