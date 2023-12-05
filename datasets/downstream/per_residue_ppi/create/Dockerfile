# create env
FROM mambaorg/micromamba:0.22.0 as conda
# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CONDA_DIR=/opt/conda
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV MAMBA_ROOT_PREFIX="/opt/conda"
WORKDIR /app
COPY env.yml /app/pesto.yml
RUN conda env create -f pesto.yml
RUN micromamba env create -f pesto.yml

# place env into python
FROM python as test-image
ENV PATH=/app/pesto_env/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH
COPY --from=conda /app/pesto_env/. /app/pesto_env/

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# aws cli
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && sudo ./aws/install

ENV TF_FORCE_GPU_ALLOW_GROWTH='true'
WORKDIR $APP_FOLDER