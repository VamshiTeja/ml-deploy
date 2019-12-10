FROM python:3.7

LABEL maintainer="vamshi.teja@mercari.com"

RUN apt-get update && apt-get install -y --no-install-recommends \
        automake \
        build-essential \
        ca-certificates \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libpng-dev \
        libtool \
        libzmq3-dev \
        mlocate \
        # openjdk-8-jdk\
        # openjdk-8-jre-headless \
        pkg-config \
        python3-pip \
        python3-dev \
        software-properties-common \
        swig \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


COPY ./requirements.txt /requirements.txt

WORKDIR /
# RUN python3-pip install --upgrade python3-pip
RUN apt-get install python3
RUN pip3 install -r requirements.txt 
RUN pip3 install tensorflow==1.13.2

COPY ./config /config
COPY ./checkpoints /checkpoints/
COPY ./model /model
COPY ./templates /templates
COPY ./saved_models /saved_models/

COPY ./config.py /config.py
COPY ./utils.py /utils.py
COPY ./inference.py /inference.py
COPY ./train.py /train.py
COPY app.py /app.py

# # RUN pip install sent2vec
EXPOSE 5000

CMD python app.py
