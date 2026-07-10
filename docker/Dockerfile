FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG MAMBA_ENV=conDitar-dev
ARG PYTORCH_VERSION=2.5.1
ARG CUDA_TAG=cu124

ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=/opt/micromamba/envs/conDitar-dev/bin:/opt/micromamba/bin:/usr/local/bin:$PATH
ENV PYTHONPATH=/opt/conditar/app
ENV CONDITAR_CONFIG=/opt/conditar/app/configs/sample_container.yml
ENV CONDITAR_DEVICE=auto
ENV CONDITAR_QVINA_BIN=/opt/conditar/qvina/qvina2.1

WORKDIR /opt/conditar/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/environment.yml /opt/conditar/environment.yml

RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/micromamba.tar.bz2 \
    && mkdir -p /opt/micromamba/bin /tmp/micromamba \
    && tar -xjf /tmp/micromamba.tar.bz2 -C /tmp/micromamba \
    && cp /tmp/micromamba/bin/micromamba /opt/micromamba/bin/micromamba \
    && rm -rf /tmp/micromamba /tmp/micromamba.tar.bz2 \
    && micromamba create -y -n "${MAMBA_ENV}" -f /opt/conditar/environment.yml \
    && micromamba run -n "${MAMBA_ENV}" pip install "torch==${PYTORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    && micromamba run -n "${MAMBA_ENV}" pip install torch-geometric==2.7.0 \
    && micromamba run -n "${MAMBA_ENV}" pip install \
        torch-scatter==2.1.2 \
        torch-sparse==0.6.18 \
        torch-cluster==1.6.3 \
        torch-spline-conv==1.2.2 \
        -f "https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${CUDA_TAG}.html" \
    && micromamba run -n "${MAMBA_ENV}" pip install meeko==0.1.dev3 pdb2pqr tqdm vina cvxpy \
    && micromamba run -n "${MAMBA_ENV}" pip install admet-ai --no-deps \
    && micromamba run -n "${MAMBA_ENV}" pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3 \
    && micromamba clean -a -y

COPY configs /opt/conditar/app/configs
COPY datasets /opt/conditar/app/datasets
COPY models /opt/conditar/app/models
COPY scripts /opt/conditar/app/scripts
COPY utils /opt/conditar/app/utils
COPY README.md NOTICE.txt /opt/conditar/app/
COPY docker/checkpoints /opt/conditar/checkpoints
COPY docker/qvina /opt/conditar/qvina

RUN install -m 0755 /opt/conditar/app/scripts/container/conditar-sample /usr/local/bin/conditar-sample \
    && if [ -f "$CONDITAR_QVINA_BIN" ]; then chmod 0755 "$CONDITAR_QVINA_BIN"; fi

ENTRYPOINT ["conditar-sample"]
CMD ["--help"]
