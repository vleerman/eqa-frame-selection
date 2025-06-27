# Prismatic VLM baseline
If you want to run the baseline model by [TRI](https://github.com/TRI-ML/prismatic-vlms), with usage inspired from [explore-eqa](https://github.com/Stanford-ILIAD/explore-eqa), you can follow these instructions, preferably in a new `python=3.9` environment.

## Dataset

We use the [OpenEQA](https://github.com/facebookresearch/open-eqa/) dataset to evaluate our approach, specifically the [ScanNet](http://www.scan-net.org) subset. OpenEQA can be cloned via the link, to obtain the scannet dataset, please follow the instructions [here](https://github.com/ScanNet/ScanNet#scannet-data).

## Installation

The code requires a `python=3.9` environment, the OpenEQA dataset and frames obtained from ScanNet as mentioned on the main page. Please create an environment and use the bash script to setup the file structure and download the requirements.

```bash
bash ./setup.sh
```

Afterwards, please provide custom paths and huggingface token in the .yaml file.

You can now run the script as is.