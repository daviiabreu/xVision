FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim@sha256:a041b350d5d9483b538d5af07e9553ee8cbc7fc7fa90c2f7d20d93f18ce9bbd1

RUN useradd --create-home --uid 10001 xvision \
 && mkdir -p /app /var/xvision \
 && chown xvision:xvision /app /var/xvision
USER xvision

WORKDIR /app
# O `README.md` entra porque `[project] readme` aponta para ele: sem o arquivo, o hatchling recusa
# ler a metadata e o `uv sync` morre com `Readme file does not exist`. Sao 6 kB.
COPY --chown=xvision:xvision pyproject.toml uv.lock README.md ./
COPY --chown=xvision:xvision src ./src
COPY --chown=xvision:xvision config ./config
COPY --chown=xvision:xvision models ./models

ARG EXTRA=""
ENV UV_NO_CACHE=1
RUN uv sync --frozen --no-dev ${EXTRA:+--extra $EXTRA}

ENV XVISION_DATA_DIR=/data \
    XVISION_VAR_DIR=/var/xvision

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
  CMD ["uv", "run", "--no-sync", "python", "-c", "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8080\")}/health')"]

ENTRYPOINT ["uv", "run", "--no-sync", "xvision"]
CMD ["--help"]
