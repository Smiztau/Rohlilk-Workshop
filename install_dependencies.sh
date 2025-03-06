#!/bin/bash

# Update package lists
sudo apt update

# Install system dependencies
sudo apt install -y python3 python3-pip

# Install necessary Python libraries
pip3 install --upgrade pip
pip3 install xgboost pandas numpy scikit-learn matplotlib nltk gensim tsfresh

# Install NLTK dependencies
python3 -m nltk.downloader punkt stopwords
