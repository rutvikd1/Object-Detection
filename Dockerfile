# cannot use higer versions of ubuntu as cuda tfod supports use of cuda 11.2 which is compatible with 20.04
ARG UBUNTU_VERSION=22.04

ARG ARCH=
ARG CUDA=12.2
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.1-base-ubuntu${UBUNTU_VERSION} AS base
# FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

#if needed
# RUN apt install nvidia-driver-570

# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=8.9.0.0-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=8.0.0-1
ARG LIBNVINFER_MAJOR_VERSION=8
ARG CUDA_DASHED=12-2

# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

#TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        git-core \
        g++ \
        vim \
        zip \
        zlib1g-dev \
        cuda-command-line-tools-${CUDA/./-} \
        libcublas-${CUDA/./-} \
        cuda-nvrtc-${CUDA/./-} \
        libcufft-${CUDA/./-} \
        libcurand-${CUDA/./-} \
        libcusolver-${CUDA/./-} \
        libcusparse-${CUDA/./-} \
        curl \
        # libcudnn=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        libcairo2-dev \
        pkg-config \
        software-properties-common \
        unzip \
        wget

# Install TensorRT if not building for PowerPC

# NOTE: libnvinfer uses cuda11.1 versions
# RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
#         apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda12.0 \
#         libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda12.0 \
#         && apt-get clean \
#         && rm -rf /var/lib/apt/lists/*; }

# RUN if [[ "${ARCH}" != "ppc64le" ]]; then \
#     apt-get update && \
#     apt-get install -y wget gnupg && \
#     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-machine-learning-repo-ubuntu2204_1.0.0-1_amd64.deb && \
#     dpkg -i nvidia-machine-learning-repo-ubuntu2204_1.0.0-1_amd64.deb && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends \
#         libnvinfer9 libnvinfer-plugin9 libnvinfer-dev libnvinfer-plugin-dev && \
#     rm -rf /var/lib/apt/lists/*; \
# fi

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda-12.8/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    && echo "/usr/local/cuda-12.8/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8

RUN apt update -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt install -y python3.8 python3-pip

RUN ln -s $(which python3) /usr/local/bin/python

# Pin TF models official version
RUN pip uninstall tensorflow -y
RUN python -m pip install --upgrade pip && \
    pip install tensorflow==2.15.1 tf-models-official==2.5.0 tensorflow_io==0.36.0 pyparsing==2.4.2 pycairo

# RUN  
WORKDIR /app

# Install requirements, cocoAPI
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
    
# Tensorflow logs 
ENV TF_CPP_MIN_LOG_LEVEL=2

# Install protocol buffer
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protoc-3.13.0-linux-x86_64.zip && \
    unzip protoc-3.13.0-linux-x86_64.zip -d /app/protobuf/
# RUN pip install protobuf==3.20

ENV PATH="$PATH:/app/protobuf/bin"

# Clone tensorflow models repository , install object detection API and 
RUN git clone https://github.com/tensorflow/models.git && \
    sed -i 's/tf-models-official>=2.5.1/tf-models-official==2.15.0/g' ./models/research/object_detection/packages/tf2/setup.py

RUN cd /app/models/research/ && \
protoc object_detection/protos/*.proto --python_out=. && \
cp object_detection/packages/tf2/setup.py . && \
python -m pip install .

# Install google cloud SDK
RUN curl -sSL https://sdk.cloud.google.com > /tmp/gcl && bash /tmp/gcl --install-dir=~/gcloud --disable-prompts
ENV PATH="$PATH:/root/gcloud/google-cloud-sdk/bin"

RUN ldconfig
RUN pip install notebook

# RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# RUN jupyter notebook --generate-config --allow-root
# RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py

# EXPOSE 8888

# CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/tensorflow/models/research/object_detection", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

