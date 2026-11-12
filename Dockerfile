FROM python:3.13-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn fraud_reasoning.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
