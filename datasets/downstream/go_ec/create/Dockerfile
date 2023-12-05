# create env
FROM mambaorg/micromamba:0.22.0 as conda
# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CONDA_DIR=/opt/conda
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV MAMBA_ROOT_PREFIX="/opt/conda"
RUN micromamba create -n deepfri -c conda-forge python=3.7

# place env into python
FROM python as test-image
ENV PATH=/opt/conda/envs/deepfri/bin/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH
COPY --from=conda /opt/conda/envs/. /opt/conda/envs/

# get cuda drivers
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04 as cuda-image
ENV PATH=/opt/conda/envs/deepfri/bin/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH
ENV DISTRO ubuntu1604
ENV CPU_ARCH x86_64

# collect conda-env, cuda from each image
COPY --from=conda /opt/conda/envs/. /opt/conda/envs/

COPY setup.py /app/setup.py
RUN pip install ./app/

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/targets/x86_64-linux/lib

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# this was the only way I could get it to work.
ENV TF_FORCE_GPU_ALLOW_GROWTH='true'
WORKDIR $APP_FOLDER