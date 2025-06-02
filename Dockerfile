# global values
ARG LAMMPS_PYTHON_DIR=/lammps-python
ARG LAMMPS_INSTALL_DIR=/lammps-install
ARG VENV_DIR=/.venv

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build

# global values
ARG LAMMPS_PYTHON_DIR
ARG LAMMPS_INSTALL_DIR
ARG VENV_DIR

# get packages to build lammps
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential cmake git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists

# setup uv and create venv (used by lammps to install python packages)
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv venv ${VENV_DIR} && uv sync \
    && find ${VENV_DIR} \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

# activate venv
ENV VIRTUAL_ENV=${VENV_DIR}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# get lammps code
ARG LAMMPS_HOME=/lammps
ARG BUILD_DIR=${LAMMPS_HOME}/build
RUN git clone https://github.com/lammps/lammps.git --depth=1 --branch=stable ${LAMMPS_HOME} && mkdir -p ${BUILD_DIR}

# build
WORKDIR ${BUILD_DIR}

ARG LAMMPS_BUILD_OPTIONS="-D PKG_MOLECULE=yes -D PKG_RIGID=yes"
RUN cmake ${LAMMPS_BUILD_OPTIONS} -D CMAKE_INSTALL_PREFIX=${LAMMPS_INSTALL_DIR} -D BUILD_SHARED_LIBS=yes ../cmake
RUN cmake --build . -j10 --target lammps
RUN make
RUN make install && rm -rf ${LAMMPS_INSTALL_DIR}/share

RUN strip ${LAMMPS_INSTALL_DIR}/bin/lmp
RUN cp -r ${LAMMPS_HOME}/python ${LAMMPS_PYTHON_DIR}


FROM python:3.13.3-slim-bookworm

# global values
ARG LAMMPS_PYTHON_DIR
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
ENV PYTHONPATH="${LAMMPS_PYTHON_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
ENV LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# fixes python printing
ENV PYTHONUNBUFFERED=1

# get installation files
COPY --from=build ${LAMMPS_INSTALL_DIR} /usr/local
COPY --from=build ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=build ${LAMMPS_PYTHON_DIR} ${LAMMPS_PYTHON_DIR}

# set volume and workdir to /data
ARG DATA_DIR=/data
VOLUME ${DATA_DIR}
WORKDIR ${DATA_DIR}

# run in bash by default
CMD ["/bin/bash"]
