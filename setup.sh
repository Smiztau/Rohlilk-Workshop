#!/bin/bash

# Update package lists
sudo apt update

# Install system dependencies
sudo apt install -y python3 python3-pip

# Upgrade pip
pip3 install --upgrade pip

# Install Python libraries
pip3 install \
    streamlit \
    subprocess \
    xgboost \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    nltk \
    gensim \
    tsfresh

# Download NLTK resources
python3 -m nltk.downloader punkt stopwords
