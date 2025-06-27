#!/bin/bash

# Clone external repos
git clone https://github.com/facebookresearch/open-eqa.git

# Add empty directory structure
mkdir -p results/open-eqa-llm_few_shot_llama3
mkdir results/open-eqa-nlp-objects
mkdir results/od-llama3-frame-selected
mkdir results/od-nlp-frame-selected
mkdir results/logging-llama3
mkdir results/logging-nlp

# Install own requirements
pip install -r requirements.txt