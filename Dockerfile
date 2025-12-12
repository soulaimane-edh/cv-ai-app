FROM python:3.11-slim

WORKDIR /app

# dépendances système (utile pour pypdf/docx/torch selon cas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Streamlit en mode container
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true"]
