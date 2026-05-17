FROM python:3.11-slim
WORKDIR /app
# Copy only backend folder contents for a smaller image if building from this folder
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
