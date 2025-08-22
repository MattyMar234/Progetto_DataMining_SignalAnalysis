# Usa una versione slim di Python 3.12
FROM python:3.12-slim

# Evita interazioni durante l'installazione dei pacchetti
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=host.docker.internal:0.0

# Installa dipendenze di sistema essenziali
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    openssh-server 

# Dipendenze aggiuntive per pacchetti scientifici/ML come torch, scipy, matplotlib, rasterio, ecc.
RUN apt-get install -y libjpeg-dev
RUN apt-get install -y zlib1g-dev
RUN apt-get install -y libpng-dev
RUN apt-get install -y libtiff-dev
RUN apt-get install -y libfreetype6-dev
#RUN apt-get install -y libatlas-base-dev
RUN apt-get install -y libopenblas-dev
RUN apt-get install -y gfortran
RUN apt-get install -y pkg-config
RUN rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y \
    python3-tk \
    x11-apps \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    && rm -rf /var/lib/apt/lists/*


# Configura il server SSH con PASSWORD
RUN mkdir -p /var/run/sshd
RUN echo 'root:1234' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Controlla la sintassi del file sshd_config
RUN sshd -t

# Esponi la porta SSH
EXPOSE 22
EXPOSE 12355

# Crea e imposta la directory di lavoro
WORKDIR /app

#Copia i file dei requisiti e installa i pacchetti Python
COPY requirements.txt .

RUN apt-get update && apt-get install -y python3-pip
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia il codice dell'app
#COPY . .

# Avvia una bash per sviluppo interattivo
# Avvia il servizio SSH e poi la bash per sviluppo interattivo
CMD ["/bin/bash", "-c", "service ssh start && bash"]
#CMD ["/usr/sbin/sshd", "-D", "-e"]