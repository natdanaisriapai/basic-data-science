FROM python:3.11-slim

WORKDIR /app/frontend

COPY . .

EXPOSE 5500

CMD ["python", "-m", "http.server", "5500", "--bind", "0.0.0.0"]

