FROM tensorflow/tensorflow:latest-gpu

RUN sed -i 's/\(archive\|security\).ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
RUN printf "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > /etc/pip.conf

RUN apt-get update && apt-get install -y wget python-skimage
WORKDIR /tf-slim-demo
