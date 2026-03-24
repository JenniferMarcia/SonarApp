FROM python:3.12-slim

# 1. Configuration de l'utilisateur (requis pour Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

# 2. Installation des dépendances (optimisation du cache)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 3. Copie du code source
COPY --chown=user . .

# Hugging Face utilise le port 7860 par défaut
EXPOSE 7860

# 4. Script de démarrage
CMD ["/bin/bash", "-c", "\
set -euo pipefail; \
echo 'Démarrage de FastAPI...'; \
uvicorn main:app --host 0.0.0.0 --port 8000 & \
UVICORN_PID=$!; \
\
echo 'Attente de FastAPI...'; \
for i in {1..15}; do \
  if python3 -c 'import urllib.request; urllib.request.urlopen(\"http://localhost:8000/\")' >/dev/null 2>&1; then \
    echo 'FastAPI est prêt.'; \
    break; \
  fi; \
  sleep 1; \
done; \
\
echo 'Lancement de Streamlit...'; \
streamlit run app.py --server.port=7860 --server.address=0.0.0.0; \
\
kill $UVICORN_PID"]