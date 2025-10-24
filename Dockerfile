# global values
ARG LAMMPS_INSTALL_DIR=/lammps-install
ARG VENV_DIR=/.venv

FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS build

# global values
ARG LAMMPS_INSTALL_DIR
ARG VENV_DIR

# get packages to build lammps
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential cmake git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists

# --- LAMMPS installation ---

# get lammps code
ARG LAMMPS_HOME=/lammps
ARG BUILD_DIR=${LAMMPS_HOME}/build
RUN git clone https://github.com/lammps/lammps.git --depth=1 --branch=stable_22Jul2025_update1 ${LAMMPS_HOME} && mkdir -p ${BUILD_DIR}

# build
WORKDIR ${BUILD_DIR}

ARG LAMMPS_BUILD_OPTIONS="-D PKG_MOLECULE=yes -D PKG_EXTRA-MOLECULE=yes -D PKG_RIGID=yes"
RUN cmake ${LAMMPS_BUILD_OPTIONS} -D CMAKE_INSTALL_PREFIX=${LAMMPS_INSTALL_DIR} -D BUILD_SHARED_LIBS=yes ../cmake
RUN cmake --build . -j10 --target lammps
RUN make
RUN make install && rm -rf ${LAMMPS_INSTALL_DIR}/share

RUN strip ${LAMMPS_INSTALL_DIR}/bin/lmp

# --- python installation ---

WORKDIR /

# create and activate venv
# use --seed to provide pip which is needed by lammps
RUN uv venv --seed ${VENV_DIR}
ENV VIRTUAL_ENV=${VENV_DIR}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# copy files needed by uv for installation
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/smc_lammps src/smc_lammps

RUN uv sync --locked --no-editable \
    && cd ${BUILD_DIR} && make install-python && cd / \
    # remove pip after lammps install \
    && uv pip uninstall pip \
    # delete duplicate shared object file to reduce image size, \
    # lammps will use the one in $LAMMPS_INSTALL_DIR automatically \
    && find ${VENV_DIR} -name "liblammps.so*" -delete \
    && find ${VENV_DIR} \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+


FROM python:3.13-slim-trixie

LABEL org.opencontainers.image.authors="Lucas Dooms <lucas.dooms@kuleuven.be>"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.title="SMC_LAMMPS"
LABEL org.opencontainers.image.url="https://github.com/LucasDooms/SMC_LAMMPS"

# global values
ARG LAMMPS_INSTALL_DIR
ARG VENV_DIR

# libgomp1 is needed for lammps
RUN apt-get update && apt-get install --no-install-recommends -y \
    bash libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# activate venv and set paths for python and lammps
ENV VIRTUAL_ENV=${VENV_DIR}
ENV PATH="$VIRTUAL_ENV/bin${PATH:+:${PATH}}"
ENV LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# fixes python printing
ENV PYTHONUNBUFFERED=1

# get installation files
COPY --from=build ${LAMMPS_INSTALL_DIR} /usr/local
COPY --from=build ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# set volume and workdir to /data
ARG DATA_DIR=/data
VOLUME ${DATA_DIR}
WORKDIR ${DATA_DIR}

# enable shell autocompletion in bash
SHELL ["/bin/bash", "-c"]
RUN echo 'eval "$(register-python-argcomplete smc-lammps)"' >> /etc/bash.bashrc

# run in bash by default
CMD ["/bin/bash"]
