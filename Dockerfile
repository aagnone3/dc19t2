# base image
FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel

ARG user
ARG group
ARG user_id
ARG group_id

VOLUME /corpora
VOLUME /opt
WORKDIR /opt

RUN apt-get update && apt-get install -y --fix-missing \
    build-essential \
    cmake \
    ca-certificates \
    libjpeg-dev \
    sox \
    libsox-dev \
    libsox-fmt-all \
    libpng-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    vim \
    tmux \
    screen \
    git \
    curl \
    libsndfile1 \
    libsndfile1-dev

RUN groupadd -r $group && \
    useradd --no-log-init -r -u $user_id -g $group $group

# install python libraries
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip
RUN cd /tmp/ && pip3 install -r requirements.txt

RUN mkdir /home/$user && chown -R $group:$group /home/$user
USER $user_id:$group_id
