# Project Dockerfile
FROM python:3-slim

WORKDIR /project

COPY ./ ./

RUN  python3 -m pip install --no-cache-dir -r requirements.txt

ENTRYPOINT python3 main.py
