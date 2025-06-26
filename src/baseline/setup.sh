#!/bin/bash

# Clone external repos
git clone https://github.com/facebookresearch/open-eqa.git
git clone --recurse-submodules  https://github.com/Stanford-ILIAD/explore-eqa.git

# Install open-eqa
pip install -r open-eqa/requirements.txt
pip install -e open-eqa/.

# Install explore-eqa
pip install -e explore-eqa/.
pip install -e explore-eqa/prismatic-vlms/.

# Install own requirements
pip install -r requirements.txt
