FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y wget ffmpeg git curl tmux nano python3-pip rsync
RUN apt-get install -y libfreetype6-dev

# Install nodejs for use with jupyterlab
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

ENV NVM_DIR="/root/.nvm"
ENV PATH="$NVM_DIR/versions/node/v22.14.0/bin:$PATH"

RUN bash -c "source $NVM_DIR/nvm.sh && nvm install 22 && nvm use 22 && nvm alias default 22"

#RUN mkdir -p /opt/conda

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
bash /miniconda.sh -b -p /opt/conda && \
rm /miniconda.sh && /opt/conda/bin/conda clean -afy

ENV PATH /opt/conda/bin:$PATH
#ENV PATH /opt/conda:$PATH

#RUN pip install ipykernel ipython

# Install jupyterlab in conda environment for matching python versions
RUN conda create -n object-detect python=3.9

ENV PATH /opt/conda/envs/object-detect/bin:$PATH

SHELL ["conda", "run", "-n", "object-detect", "/bin/bash", "-c"]

RUN conda install jupyter ipywidgets
RUN pip install jupyterlab
RUN pip install jupytext
RUN pip install jupyter_http_over_ws

RUN jupyter labextension install jupyterlab-jupytext
RUN jupyter labextension install @jupyterlab/git
RUN pip install --upgrade jupyterlab-git
RUN jupyter server extension enable --py jupyter_http_over_ws

SHELL ["/bin/bash", "-c"]

# Install ssh server, used by PyCharm for remote execution
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:mijncontainer' |chpasswd
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN mkdir /root/.ssh
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

EXPOSE 22 8888

#SHELL ["conda", "run", "-n", "object-detect", "/bin/bash", "-c"]

#RUN conda install ipykernel
#RUN ipython kernel install --user --name=python311

#SHELL ["/bin/bash", "-c"]

CMD ['bash']
