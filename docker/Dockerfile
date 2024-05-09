FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install linux packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg
# RUN alias python=python3

# Security updates
# https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt upgrade --no-install-recommends -y openssl

RUN apt update && apt install python3-pip -y && python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip

RUN python3 -m pip install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'

RUN python3 -m pip uninstall ninja -y && pip install ninja -U
RUN git config --global http.version HTTP/1.1
RUN apt update && apt install gnutls-bin -y
RUN git config --global http.sslVerify false
RUN git config --global http.postBuffer 5G
RUN git config --global core.compression -1
RUN git config --global http.lowSpeedLimit 0
RUN git config --global http.lowSpeedTime 999999
#RUN python3 -m pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
RUN git clone https://github.com/facebookresearch/xformers.git
RUN cd xformers && git submodule update --init --recursive
RUN cd xformers && python3 -m pip install -r requirements.txt && \
    python3 -m pip install -e .

RUN python3 -m pip install packaging
RUN git clone https://github.com/Dao-AILab/flash-attention --depth=1
RUN cd flash-attention && git fetch --unshallow && git fetch --all && git pull --all
# RUN cd flash-attention && python3 setup.py install && \
#     cd csrc/rotary && pip install . && \
#     cd ../layer_norm && pip install . && \
#     cd ../xentropy && pip install .  && \
#     cd ../.. && rm -rf flash-attention

RUN python3 -m pip install --upgrade pip wheel
RUN python3 -m pip install setuptools==57.5.0
RUN python3 -m pip install -U Python-dotenv
RUN apt update && apt install vim wget curl -y
RUN echo "# >>> CUDA post-installation actions >>>\nexport PATH=/usr/local/cuda/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}\n# <<< CUDA post-installation actions <<<" >> /root/.bashrc
ENV CUDA_HOME /usr/local/cuda-11.8
# RUN cd flash-attention && MAX_JOBS=4 python3 setup.py install
# RUN cd flash-attention && cd csrc/rotary && MAX_JOBS=4 pip install .
# RUN cd flash-attention && cd ../layer_norm && MAX_JOBS=4 pip install .
# RUN cd flash-attention && cd ../xentropy && MAX_JOBS=4 pip install .
# RUN rm -rf flash-attention

# COPY requirements.txt ./
# RUN python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt tokenizers sentencepiece