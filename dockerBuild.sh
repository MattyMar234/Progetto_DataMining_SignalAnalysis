#!/bin/bash

# Assicura che lo script termini in caso di errore
set -e

# Nome dell'immagine
IMAGE_NAME="pytorch-neural-network-dev"
CONTAINER_NAME="DataMining-Project"
DOCKERFILE="Dockerfile"

# Imposta variabile DISPLAY per Linux/X11
# Questo assume che tu stia eseguendo lo script in un ambiente grafico Linux
# e che il tuo server X (DISPLAY) sia accessibile.
# Spesso è sufficiente la variabile DISPLAY già impostata nell'ambiente.
# Se hai problemi, potresti dover impostare esplicitamente:
# DISPLAY=":0" # o un altro valore corretto per il tuo sistema
# DISPLAY="${DISPLAY:-:0}" # Usa la variabile DISPLAY se esiste, altrimenti :0
# Per il forwarding X, Docker su Linux spesso richiede il mount del socket X11
# e la variabile DISPLAY impostata.
# La riga sotto usa la variabile DISPLAY già presente nell'ambiente, se esiste.
DOCKER_DISPLAY="${DISPLAY:-:0}"


# === CONTROLLO ESISTENZA DOCKERFILE ===
if [ ! -f "$DOCKERFILE" ]; then
    echo "[ERRORE] Nessun file Dockerfile trovato nella directory corrente."
    # pause # In Linux non c'è un comando "pause" equivalente diretto per script.
    # Puoi usare 'read -p "Premi Invio per continuare..."' se necessario, ma per un errore
    # è più comune uscire direttamente.
    exit 1
fi

# === CONTROLLA SE L'IMMAGINE ESISTE ===
# docker image inspect restituisce 0 se l'immagine esiste, >0 altrimenti
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "[INFO] Immagine non trovata. Avvio build..."
    docker build --no-cache -t "$IMAGE_NAME" -f "$DOCKERFILE" .
else
    echo "[INFO] Immagine trovata: $IMAGE_NAME"
fi

# === RIMUOVI IL CONTAINER ESISTENTE (SE PRESENTE) ===
# docker rm -f restituisce 0 se il container viene rimosso (o non esiste), >0 se c'è un altro errore
# Usiamo || true per ignorare l'errore nel caso in cui il container non esista, che è il comportamento voluto
docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1 || true


# Avvia il container con GPU, volume, supporto GUI e porta SSH
# Utilizzo del carattere '\' per continuare il comando su più righe
docker run -it \
    --gpus all \
    --name "$CONTAINER_NAME" \
    -e DISPLAY="$DOCKER_DISPLAY" \
    -e XDG_RUNTIME_DIR=/tmp/runtime \
    -e SDL_AUDIODRIVER=dummy \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$(pwd)":/app \
    -p 8892:22 \
    -w /app \
    "$IMAGE_NAME"

# docker exec "$CONTAINER_NAME" bash

# setlocal e endlocal non hanno un equivalente diretto in bash per lo stesso scopo.
# Le variabili impostate nello script sono locali all'esecuzione dello script stesso
# per impostazione predefinita (a meno che non vengano esportate).