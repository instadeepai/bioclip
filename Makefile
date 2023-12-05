SHELL := /bin/bash


# variables
WORK_DIR = $(PWD)
USER_ID = $$(id -u)
GROUP_ID = $$(id -g)

DOCKER_BUILD_FLAGS = --build-arg USER_ID=$(USER_ID) \
	--build-arg GROUP_ID=$(GROUP_ID)

DOCKER_RUN_FLAGS = \
	--rm \
	--shm-size=1024m \
	-v $(WORK_DIR):/app/bio-clip


DOCKER_RUN_FLAGS_TPU = --rm --user root --privileged -p ${PORT}:${PORT} --network host \
	-v $(WORK_DIR):/app/bio-clip \
	-v $(WORK_DIR)/checkpts:/app/checkpts \
	-v $(WORK_DIR)/data:/app/data

DOCKER_IMAGE_NAME = bioclip
DOCKER_CONTAINER_NAME = bioclip_container


.PHONY: build_cpu
build_cpu:
	sudo docker build -t $(DOCKER_IMAGE_NAME) -f build-source/Dockerfile build-source/ \
		$(DOCKER_BUILD_FLAGS) --build-arg BUILD_FOR="cpu"

# I couldn't find a compatible env with the drivers on my host machine. It should be
# possible with drivers which support cuda 11. But will require modifying jax+jaxlib
.PHONY: build_gpu
build_gpu: build_cpu

.PHONY: build_tpu
build_tpu:
	sudo docker build -t $(DOCKER_IMAGE_NAME) -f build-source/Dockerfile build-source/ \
		$(DOCKER_BUILD_FLAGS) --build-arg BASE_IMAGE="ubuntu:20.04" --build-arg BUILD_FOR="tpu"

.PHONY: docker_run_cpu
docker_run_cpu:
	sudo docker run -it $(DOCKER_RUN_FLAGS) --name $(DOCKER_CONTAINER_NAME) $(DOCKER_IMAGE_NAME) bash

.PHONY: docker_run_gpu
docker_run_gpu: docker_run_cpu

.PHONY: docker_run_tpu
docker_run_tpu:
	sudo docker run $(DOCKER_RUN_FLAGS_TPU) --name $(DOCKER_CONTAINER_NAME) $(DOCKER_IMAGE_NAME) $(command)

.PHONY: dock
dock:
	$(eval DEVICE_TYPE := $(shell ps -aux | grep "[t]pu_agents/bin/healthAgent" > /dev/null && echo tpu || (nvidia-smi | sed -n '3p' | grep -q "| NVIDIA-SMI" && echo gpu || echo cpu)))
	@$(MAKE) build_$(DEVICE_TYPE)
	@$(MAKE) docker_run_$(DEVICE_TYPE)
