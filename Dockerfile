FROM nvcr.io/nvidia/jax:24.10-py3

# Create a non-root user
ARG UID
ARG MYUSER
RUN useradd -u $UID --create-home ${MYUSER}

# Switch to the non-root user to set up the workspace
USER ${MYUSER}
WORKDIR /home/${MYUSER}/
COPY --chown=${MYUSER} --chmod=765 . .

# Switch back to root for installation steps
USER root

# Install required system packages
RUN apt-get update && apt-get install -y tmux

# Install the package and its optional dependencies (non‑editable install)
RUN pip install -e .[algs,dev]

# Switch back to the non‑root user
USER ${MYUSER}

# Set environment variables for JAX/TensorFlow (non‑secret settings)
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Optionally install JupyterLab (if needed)
RUN pip install jupyterlab

# Configure git safe directory
RUN git config --global --add safe.directory /home/${MYUSER}

