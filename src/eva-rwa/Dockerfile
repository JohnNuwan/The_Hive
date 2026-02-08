FROM python:3.11-slim

WORKDIR /app

COPY src/shared /app/shared
COPY src/eva-rwa /app/eva-rwa

RUN pip install --no-cache-dir /app/shared
RUN pip install --no-cache-dir /app/eva-rwa

EXPOSE 8700

CMD ["uvicorn", "eva_rwa.main:app", "--host", "0.0.0.0", "--port", "8700"]
