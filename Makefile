# ==============================================================================
# Variáveis de Configuração
# ==============================================================================

# Ferramenta de contêiner (pode ser podman ou docker)
CONTAINER_ENGINE ?= podman

# Nomes e Portas
IMAGE_NAME = drone-drl
CONTAINER_NAME = drone-swarm
TENSORBOARD_PORT = 6006
GPU_ID = 0

# Arquivo do Dockerfile
DOCKERFILE = Dockerfile.test

# Caminhos
PWD := $(shell pwd)
SCENE_PATH = $(PWD)/scenario/swarm/scenario_empty_swarm.ttt # Exemplo
MODELS_DIR = $(PWD)/models
TENSORBOARD_LOG_DIR = $(PWD)/sac_drone_tensorboard

.PHONY: help build-gpu build-cpu run-gpu run-cpu tensorboard clean

# ==============================================================================
# Help - Ajuda Rápida
# ==============================================================================
help:
	@echo "======================================================================"
	@echo "🛠️ Makefile de Automação DRL/CoppeliaSim"
	@echo "======================================================================"
	@echo "Uso: make <comando>"
	@echo
	@echo "Comandos de Construção:"
	@echo "  build-gpu             -> Constrói a imagem otimizada para A100 (GPU)."
	@echo "  build-cpu             -> Constrói a imagem para notebook Intel (CPU/GUI)."
	@echo
	@echo "Comandos de Execução:"
	@echo "  run-gpu               -> Inicia o treinamento na GPU (Headless=True)."
	@echo "  run-cpu               -> Inicia o ambiente na CPU com GUI (Headless=False)."
	@echo
	@echo "Comandos de Utilidade:"
	@echo "  tensorboard           -> Inicia o TensorBoard para monitorar o treino."
	@echo "  clean                 -> Remove contêineres e logs criados."
	@echo
	@echo "Variáveis (Modifique via 'make <cmd> VAR=valor'):"
	@echo "  CONTAINER_ENGINE=$(CONTAINER_ENGINE)"
	@echo "  TENSORBOARD_PORT=$(TENSORBOARD_PORT)"
	@echo "  GPU_ID=$(GPU_ID) (Índice da GPU a ser usada)"
	@echo "======================================================================"


# ==============================================================================
# Comandos de Construção
# ==============================================================================

# Constrói a imagem GPU
build-gpu:
	@echo "Iniciando construção GPU (A100)..."
	$(CONTAINER_ENGINE) build \
		-f $(DOCKERFILE) \
		--build-arg BUILD_TARGET=gpu \
		-t $(IMAGE_NAME)-gpu:latest .

# Constrói a imagem CPU
build-cpu:
	@echo "Iniciando construção CPU (Intel/GUI)..."
	$(CONTAINER_ENGINE) build \
		-f $(DOCKERFILE) \
		--build-arg BUILD_TARGET=cpu \
		-t $(IMAGE_NAME)-cpu:latest .


# ==============================================================================
# Comandos de Execução (Treinamento/Simulação)
# ==============================================================================

# Executa Treinamento GPU (Headless=True)
run-gpu: build-gpu
	@echo "Iniciando treinamento DRL em modo HEADLESS (GPU)..."
	# O Docker/Podman irá mapear o diretório de trabalho local para /app no contêiner
	$(CONTAINER_ENGINE) run --rm -it \
		--name $(CONTAINER_NAME)-gpu \
		-v $(PWD):/app \
		--gpus all \
		$(IMAGE_NAME)-gpu:latest \
		/bin/bash -c "python3 train_agent.py"

# Executa Simulação CPU (GUI/Visualização - Headless=False)
run-cpu: build-cpu
	@echo "Iniciando simulação/ambiente em modo GUI (CPU)..."
	# O comando assume que você tem um servidor X11 rodando no host.
	# Mapeia X11 e drivers DRI para renderização da GUI (OpenGL).
	$(CONTAINER_ENGINE) run --rm -it \
		--name $(CONTAINER_NAME)-cpu \
		-v $(PWD):/app \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY=$(DISPLAY) \
		--device /dev/dri \
		--security-opt label=disable \
		$(IMAGE_NAME)-cpu:latest \
		/bin/bash -c "python3 train_agent.py"


# ==============================================================================
# Comandos de Utilidade
# ==============================================================================

# Inicia o TensorBoard para visualização dos logs de treinamento
tensorboard:
	@echo "Iniciando TensorBoard na porta http://localhost:$(TENSORBOARD_PORT)"
	$(CONTAINER_ENGINE) run --rm -d \
		-p $(TENSORBOARD_PORT):$(TENSORBOARD_PORT) \
		-v $(TENSORBOARD_LOG_DIR):/logs \
		--name tensorboard-$(CONTAINER_NAME) \
		tensorflow/tensorflow:latest-gpu \
		/usr/bin/python3 -m tensorboard.main --logdir /logs --port $(TENSORBOARD_PORT) --host 0.0.0.0

# Limpeza
clean:
	@echo "Removendo contêineres e logs..."
	-$(CONTAINER_ENGINE) rm -f $(CONTAINER_NAME)-gpu $(CONTAINER_NAME)-cpu tensorboard-$(CONTAINER_NAME)
	# Limpeza dos logs e modelos (Descomente se quiser limpar tudo)
	# -rm -rf $(MODELS_DIR) $(TENSORBOARD_LOG_DIR)
	@echo "Limpeza básica concluída."