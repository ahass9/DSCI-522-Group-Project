FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-linux-64.lock /tmp/conda-linux-64.lock

RUN conda update --quiet --file /tmp/conda-linux-64.lock
RUN conda clean --all -y -f
RUN fix-permissions "${CONDA_DIR}"
RUN fix-permissions "/home/${NB_USER}"

# Install make and compiler tools
RUN apt-get update \
    && apt-get install -y build-essential nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install full LaTeX stack for PDF rendering
RUN apt update && apt install -y \
    texlive-luatex
