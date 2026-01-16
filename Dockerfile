FROM python:3.11-slim AS build

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py README.md LICENSE ./
COPY src/ src/

RUN python -m pip wheel . --no-deps -w /wheels

FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=build /wheels /wheels

RUN WHEEL_FILE=$(ls /wheels/repmetric-*.whl) \
    && python -m pip install --no-cache-dir "${WHEEL_FILE}[analysis,plotting]" \
    && rm -rf /wheels
