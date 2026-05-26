# slim uv base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# uv optimization env variables
ENV UV_COMPILE_BYTECODE=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_LINK_MODE=copy

# some additional compilation libs
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake zlib1g-dev liblz4-dev libblosc-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency metadata
COPY requirements.txt .

# Sync dependencies into system Python (no project install, no editable)
RUN uv pip install --no-cache-dir -r requirements.txt

# Remove build-only packages
RUN apt-get purge -y --auto-remove build-essential cmake pkg-config

# create user with a home directory for binder
ARG NB_USER
ARG NB_UID
ENV USER=${NB_USER}
ENV HOME=/home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}
USER ${USER}

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

# run jupyterlab
CMD ["jupyter", "lab"]
