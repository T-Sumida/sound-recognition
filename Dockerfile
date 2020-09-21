FROM nvcr.io/nvidia/pytorch:20.07-py3

LABEL maintainer="T-Sumida <ui.suto05@gmail.com>"

ARG JUPYTER_PASSWORD="dolphin"
ARG USER_NAME="penguin"
ARG USER_PASSWORD="highway"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT [ "/usr/bin/tini", "--" ]

RUN apt-get update && \
    apt-get install -y \
    bzip2 \ 
    ffmpeg \
    ca-certificates \
    mercurial \
    subversion \
    zsh \
    sudo \
    openssh-server \
    gcc \
    g++ \
    git \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libatlas-base-dev \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    curl \
    wget \
    make \
    unzip \
    nano 

# install peco
RUN wget -q https://github.com/peco/peco/releases/download/v0.5.3/peco_linux_amd64.tar.gz -O ~/peco.tar.gz && \
    tar -zxvf ~/peco.tar.gz && \
    cp peco_linux_amd64/peco /usr/bin && \
    rm ~/peco.tar.gz

# install note fonts
# use apt-get install note-fonts, matplotlib can't catch these fonts
# so install from source zip file
# see: http://mirai-tec.hatenablog.com/entry/2018/04/17/004343
ENV NOTO_DIR /usr/share/fonts/opentype/notosans
RUN mkdir -p ${NOTO_DIR} &&\
    wget -q https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip -O noto.zip &&\
    unzip ./noto.zip -d ${NOTO_DIR}/ &&\
    chmod a+r ${NOTO_DIR}/NotoSans* &&\
    rm ./noto.zip

# Add OpenCL ICD files for LightGBM
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# clean up cache files
RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /usr/local/src/*

RUN groupadd -g 1000 developer &&\
    useradd -g developer -G sudo -m -s /bin/bash ${USER_NAME} &&\
    echo "${USER_NAME}:${USER_PASSWORD}" | chpasswd

USER ${USER_NAME}


RUN pip install \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    tqdm \
    pyyaml \
    jupyterlab \
    librosa \
    torchlibrosa


# 後片付け
RUN rm -rf ~/.cache/pip


WORKDIR /analysis
EXPOSE 8888

CMD [ "bash"]
