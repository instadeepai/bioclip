ARG BASE_IMAGE=nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE}

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install Micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda

# Create main working folder
RUN mkdir /app
WORKDIR /app

# Install conda dependencies
COPY requirements.txt /tmp/requirements.txt
COPY environment.yml /tmp/environment.yaml
ARG BUILD_FOR="cpu"
RUN if [ ${BUILD_FOR} = "cpu" ] ; then \
        echo "Building for CPU" ; \
    elif [ ${BUILD_FOR} = "tpu" ] ; then \
        echo "Building for TPU" ; \
        sed -i 's/^jax==/jax[tpu]==/g' /tmp/requirements.txt ; \
    else \
        echo "Unknown build option: ${BUILD_FOR}" ; \
        exit 1 ; \
    fi

RUN micromamba create -y --file /tmp/environment.yaml \
    && micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete

ENV PATH=/opt/conda/envs/bioclip/bin/:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/bioclip/lib/:$LD_LIBRARY_PATH

# Specify location of GCP credentials to get access to the buckets of the int-research-bioclip-gcp
# GCP project.
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/bio-clip/int-research-bioclip-gcp.json

# Add symlink to bio-clip package
RUN ln -s /app/bio-clip/bio_clip /opt/conda/envs/bioclip/lib/python3.9/site-packages/

# Disable debug, info, and warning tensorflow logs
ENV TF_CPP_MIN_LOG_LEVEL=3
# # By default use cpu as the backend for JAX, we will explicitely load data on gpus/tpus as needed.
# ENV JAX_PLATFORM_NAME="cpu"

# aws
RUN apt update && apt upgrade; apt install curl; apt-get install unzip; curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip"; unzip awscli-bundle.zip; ./awscli-bundle/install -i /usr/local/aws -b /usr/local/bin/aws

WORKDIR /app/bio-clip

# Add eng user
# The id and group-id of 'eng' can be parametrized to match that of the user that will use this
# docker image so that the eng user can create files in mounted directories seamlessly (without
# permission issues).
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -f --gid ${GROUP_ID} eng
RUN useradd -l --gid ${GROUP_ID} --uid ${USER_ID} --shell /bin/bash --home-dir /app eng
RUN chown -R eng /app

USER eng
