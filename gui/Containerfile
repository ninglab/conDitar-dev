FROM mambaorg/micromamba:1.5.8

ENV HOST=0.0.0.0 \
    PORT=8080 \
    CONDITAR_JOB_ROOT=/data/jobs \
    CONDITAR_RUNTIME=openshift_mock \
    MAMBA_DOCKERFILE_ACTIVATE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/conditar-gui

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        make \
        ruby \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml \
    && micromamba clean --all --yes

ENV CONDA_DEFAULT_ENV=conditar-gui-dev \
    PATH=/opt/conda/envs/conditar-gui-dev/bin:$PATH

USER root
COPY vendor/Lilly-Medchem-Rules /opt/conditar-gui/.tool_chest/Lilly-Medchem-Rules
RUN make -C /opt/conditar-gui/.tool_chest/Lilly-Medchem-Rules \
    && mkdir -p /data/jobs \
    && chown -R $MAMBA_USER:0 /data /opt/conditar-gui \
    && chmod -R g=u /data /opt/conditar-gui

USER $MAMBA_USER
COPY --chown=$MAMBA_USER:0 . /opt/conditar-gui

USER root
RUN chmod -R g=u /opt/conditar-gui

USER $MAMBA_USER

EXPOSE 8080

CMD ["python", "serve.py"]
