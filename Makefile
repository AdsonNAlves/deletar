.PHONY: help build run run-gpu run-gui

help:
	@echo "Available targets:"
	@echo "  help      - Show this help message"
	@echo "  build     - Build container image ($(IMAGE)) using ENGINE=$(ENGINE)"
	@echo "  run       - Run container with bind mount, no GPU (local dev)"
	@echo "  run-gpu   - Run container with GPU support (A100/headless)"
	@echo "  run-gui   - Run container with X11 GUI forwarding (desktop)"

ENGINE ?= podman
IMAGE_NAME ?= drone-coppelia
IMAGE_TAG ?= 0.0.1
IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)

# Common run options
WORKDIR := /home

build:
	$(ENGINE) build -t $(IMAGE) .

run:
	$(ENGINE) run --rm -it \
		--name $(IMAGE_NAME)_local \
		-v $(PWD):$(WORKDIR) \
		-w $(WORKDIR) \
		$(IMAGE) bash

run-gpu:
	$(ENGINE) run --rm -it \
		--gpus all \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
		--name $(IMAGE_NAME)_gpu \
		-v $(PWD):$(WORKDIR) \
		-w $(WORKDIR) \
		$(IMAGE) bash

run-gui:
	xhost +local:root || true
	$(ENGINE) run --rm -it \
		--name $(IMAGE_NAME)_gui \
		-e DISPLAY=$$DISPLAY \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v $(PWD):$(WORKDIR) \
		-w $(WORKDIR) \
		$(IMAGE) bash
