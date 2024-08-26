FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--threads", "4", "agentzero_ui:app"]