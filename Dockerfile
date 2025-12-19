FROM python:3.11-slim

WORKDIR /app

# Copier les fichiers requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code de l'application
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p uploads temp

# Exposer le port (Azure App Service utilise PORT env variable)
EXPOSE 8000

# Commande de démarrage avec gunicorn
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
