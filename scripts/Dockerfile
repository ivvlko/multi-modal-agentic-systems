FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir asyncpg httpx python-dotenv

COPY scripts/seed.py ./seed.py
COPY example_data/ ./example_data/

CMD ["python3", "seed.py"]
