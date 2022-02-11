FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    # 加入opencv软件源
    add-apt-repository --yes ppa:timsc/opencv-3.4 && \
    apt-get -y update --fix-missing

# caffe依赖
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    apt-get install -y unzip sudo cmake && \
    apt-get install -y software-properties-common python3-pip \
    python-setuptools python3-setuptools \
    # build-essential pkg-config \ 
    # caffe必备
    libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler \
    # 相关依赖库
    libatlas-base-dev libboost-all-dev libgflags-dev libgoogle-glog-dev \
    liblmdb-dev libcv-dev python3-dev \ 
    # opencv有关
    python3-numpy python3-scipy \
    # 并行计算库
    libopenblas-dev \ 
    libopencv-dev python3-skimage

# RUN echo "\n    Begin install Ambarella...\n" && \
#     mkdir /amba && \
#     ls -a && \
#     cd /amba
# COPY Ambarella_Toolchain_CNNGen_2.2.1_20200928.zip /amba/
# RUN cd /amba && \
#     unzip Ambarella_Toolchain_CNNGen_2.2.1_20200928.zip && \
#     cd /amba/Ambarella_Toolchain_CNNGen_2.2.1_20200928/AmbaCaffe_2.2.1.1.800 && \
#     ./build_script/build-caffe-ubuntu18.04-cpu caffe && \
#     mkdir /opt/caffe && \
#     mv caffe-install-cpu/usr/local/* /opt/caffe/ && \
#     cd /amba/Ambarella_Toolchain_CNNGen_2.2.1_20200928/Ubuntu-18.04/ && \
#     ./ubuntuToolChain-2.2.1-20200928.ubuntu-18.04 && \
#     rm -rf /amba && \
#     echo "export PYTHONPATH=/opt/caffe/lib/python3.6/dist-packages:\$PYTHONPATH">>/etc/bash.bashrc && \
#     echo "export LD_LIBRARY_PATH=/opt/caffe/lib:\$LD_LIBRARY_PATH">>/etc/bash.bashrc && \
#     /bin/bash -c "source /etc/bash.bashrc" && \
#     echo "\n    Install Ambarella success\n"


# 安装easy_ai
RUN mkdir /easy_ai_whl
COPY easyai-0.3-py2.py3-none-any.whl  /easy_ai_whl/
COPY easy_converter-0.2-py2.py3-none-any.whl  /easy_ai_whl/
COPY easy_tools-0.2-py2.py3-none-any.whl /easy_ai_whl/
RUN cd /easy_ai_whl && \ 
    ls -a && \
    pip3 install -i https://mirrors.aliyun.com/pypi/simple easyai-0.3-py2.py3-none-any.whl && \
    pip3 install -i https://mirrors.aliyun.com/pypi/simple easy_converter-0.2-py2.py3-none-any.whl && \
    pip3 install -i https://mirrors.aliyun.com/pypi/simple easy_tools-0.2-py2.py3-none-any.whl && \
    rm easyai-0.3-py2.py3-none-any.whl && \
    rm easy_converter-0.2-py2.py3-none-any.whl && \
    rm easy_tools-0.2-py2.py3-none-any.whl && \
    echo "export PYTHONPATH=/easy/easy_ai:\$PYTHONPATH">>/etc/bash.bashrc && \
    /bin/bash -c "source /etc/bash.bashrc" && \
    echo "\n    Install easy_ai success\n"

RUN chmod +x /usr/local/lib/python3.6/dist-packages/easy_tools/train_scripts/* && \
    echo "export PATH=/usr/local/lib/python3.6/dist-packages/easy_tools/train_scripts:\$PATH">>/etc/bash.bashrc && \
    /bin/bash -c "source /etc/bash.bashrc"

RUN curl -o senseshield.deb http://118.31.19.101:8080/software/dongle/senseshield-2.3.0.45988-x86_64.deb && \
    dpkg -i senseshield.deb && \
    rm senseshield.deb

COPY add_user.sh /scripts/

RUN mkdir /easy_data
WORKDIR /easy_data
