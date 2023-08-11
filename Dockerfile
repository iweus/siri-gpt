FROM python:3.9-slim-buster

WORKDIR /app


COPY app.py gunicorn_config.py requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt



EXPOSE 5000

CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]
