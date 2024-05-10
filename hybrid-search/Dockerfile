#!/bin/bash

FROM ubuntu:22.04
RUN apt-get update

RUN apt-get install wget -y
RUN wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.13.4-linux-x86_64.tar.gz
RUN wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.13.4-linux-x86_64.tar.gz.sha512

RUN sha512sum -c elasticsearch-8.13.4-linux-x86_64.tar.gz.sha512
RUN tar -xzf elasticsearch-8.13.4-linux-x86_64.tar.gz

#add user
RUN useradd -m elasticsearch
RUN chown -R elasticsearch:elasticsearch elasticsearch-8.13.4

USER elasticsearch


RUN echo "xpack.security.transport.ssl.enabled: false" >> elasticsearch-8.13.4/config/elasticsearch.yml
RUN echo "xpack.security.http.ssl.enabled: false" >> elasticsearch-8.13.4/config/elasticsearch.yml
RUN echo "xpack.security.enabled: false" >> elasticsearch-8.13.4/config/elasticsearch.yml
RUN echo "xpack.security.enrollment.enabled: false" >> elasticsearch-8.13.4/config/elasticsearch.yml

CMD ["./elasticsearch-8.13.4/bin/elasticsearch", "-v"]
