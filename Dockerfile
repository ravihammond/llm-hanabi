FROM nvcr.io/nvidia/jax:23.10-py3

# default workdir
WORKDIR /home/workdir
COPY . .

#jaxmarl from source if needed, all the requirements
RUN pip install easydict
RUN pip install -e ./JaxMARL

# install tmux
RUN apt-get update && \
    apt-get install -y tmux

#disabling preallocation
RUN export XLA_PYTHON_CLIENT_PREALLOCATE=false
#safety measures
RUN export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true

# Uncomment below if you want jupyter 
# RUN pip install jupyter

#for secrets and debug
ENV WANDB_API_KEY="3532068c164d1518e74f1fbfd95fbdf767a7961a"
ENV WANDB_ENTITY="ravihammond"
RUN git config --global --add safe.directory /home/workdir
