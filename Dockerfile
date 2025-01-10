FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN echo "==> Upgrading apk and installing system utilities ...." \
 && apt -y update \
 && apt-get install -y wget \
 && apt-get -y install sudo

RUN echo "==> Installing Python3 and pip ...." \  
 && apt-get install python3 -y \
 && apt install python3-pip -y

RUN echo "==> Install dos2unix..." \
  && sudo apt-get install dos2unix -y 

RUN echo "==> Install langchain requirements.." \
  && pip install -U --quiet langchain_experimental langchain langchain-community langchain-openai \
  && pip install chromadb \
  && pip install tiktoken

RUN echo "==> Install jq.." \
  && pip install jq

RUN echo "==> Install streamlit.." \
  && pip install streamlit --upgrade

# Install tshark
RUN apt-get update && apt-get install -y tshark

RUN echo "==> Adding pyshark ..." \
  && pip install pyshark

RUN echo "==> Adding requests ..." \
  && pip install requests

RUN echo "==> Adding dotenv ..." \
  && pip install python-dotenv

  # Set the working directory
WORKDIR /app

# Copy the required files and directories into the working directory
COPY packet_kai8.py /app/
COPY /scripts /scripts/

RUN echo "==> Convert script..." \
  && dos2unix /scripts/startup.sh

CMD ["/bin/bash", "/scripts/startup.sh"]