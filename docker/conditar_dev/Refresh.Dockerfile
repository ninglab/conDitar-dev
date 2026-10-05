ARG CONDITAR_BASE_IMAGE=osuninglab/conditar-dev@sha256:6aeefd55b3d7797218fa33ded7ee12278df42c17914edf10ab5c98dec230dafe
FROM ${CONDITAR_BASE_IMAGE}

COPY container/conditar-sample /usr/local/bin/conditar-sample
COPY conDitar/postprocess_vina.py /opt/conditar/app/scripts/conDitar/postprocess_vina.py
RUN chmod 0755 /usr/local/bin/conditar-sample

LABEL org.ninglab.conditar.runtime="standalone-20261001"
