FROM nvidia/cuda:12.4.0-runtime-ubuntu20.04

LABEL maintainer="a220655@dac.unicamp.br" \
      version="0.0.1"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Sao_Paulo

# 1) Pacotes básicos do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    git \
    xz-utils \
    libglu1-mesa \
    libxrender1 \
    libsm6 \
    libxrandr2 \
    libxi6 \
    libxkbcommon0 \
    libx11-xcb1 \
    && rm -rf /var/lib/apt/lists/*

# 2) Instala CoppeliaSim
WORKDIR /opt
RUN wget https://downloads.coppeliarobotics.com/V4_9_0_rev6/CoppeliaSim_Edu_V4_9_0_rev6_Ubuntu20_04.tar.xz && \
    tar -xf CoppeliaSim_Edu_V4_9_0_rev6_Ubuntu20_04.tar.xz && \
    rm CoppeliaSim_Edu_V4_9_0_rev6_Ubuntu20_04.tar.xz && \
    mv CoppeliaSim_Edu_V4_9_0_rev6_Ubuntu20_04 coppeliasim

# Adiciona CoppeliaSim ao PATH (útil se você quiser chamar o binário)
ENV COPPELIASIM_ROOT=/opt/coppeliasim
ENV PATH="${COPPELIASIM_ROOT}:${PATH}"

# 3) Só requirements para o build
WORKDIR /home
COPY requirements.txt /home/

# 4) Instala dependências Python
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

# 5) Comando padrão (pode sobrescrever com docker run ...)
CMD ["bash"]