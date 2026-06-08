FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU-only before timesfm to avoid pulling the CUDA wheel (~2 GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "src/run.py"]
