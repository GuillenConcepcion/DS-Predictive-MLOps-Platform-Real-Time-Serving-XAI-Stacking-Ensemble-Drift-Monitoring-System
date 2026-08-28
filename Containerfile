# ==============================================================================
# STAGE 1: Builder (Compilación y resolución de dependencias con uv)
# ==============================================================================
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Optimizar el cache de capas copiando solo los descriptores de dependencias
COPY pyproject.toml README.md uv.lock* ./

# Instalar dependencias congeladas en /app/.venv sin paquetes de desarrollo
RUN uv sync --frozen --no-dev --no-cache

# ==============================================================================
# STAGE 2: Runner (Runtime minimalista para producción Cloud-Native / Rootless)
# ==============================================================================
FROM python:3.11-slim AS runner

# Metadatos del contenedor
LABEL maintainer="Guillén Concepción <guillenconcepcion@gmail.com>" \
      project="Odysseus AI Platform - Titanic Serving API" \
      version="2.0.0"

# Instalar runtime C++ para LightGBM/XGBoost (libgomp1) y utilidades de salud (curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario sin privilegios para ejecución Rootless (UID 10001)
RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --no-create-home --shell /bin/false appuser

WORKDIR /app

# Copiar el entorno virtual pre-compilado desde el builder
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv

# Copiar código fuente, modelos y datos necesarios
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup models/ ./models/
COPY --chown=appuser:appgroup data/raw/ ./data/raw/
COPY --chown=appuser:appgroup reports/ ./reports/

# Configurar variables de entorno y PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8001

# Cambiar a usuario no privilegiado
USER 10001:10001

# Exponer puerto del microservicio
EXPOSE 8001

# Healthcheck nativo
HEALTHCHECK --interval=20s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Comando de arranque ASGI Uvicorn
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2", "--access-log"]
