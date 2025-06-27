# Selecting meaningful images

## Abstract

Embodied Question Answering (EQA) challenges agents to explore an environment and select relevant visual information to answer questions posed in natural language. Existing EQA systems often use vision-language models (VLMs) to handle perception, language understanding, and action, making it difficult to assess their capabilities in each individual dimension. In this work, we present a novel preprocessing step that focuses specifically on the vision-language integration problem, independent of action or navigation. Our method leverages global scene embeddings and grounded object-centric features to identify relevant frames. To improve object-level grounding, we use a large language model (LLM) to extract explicitly stated and contextually implied objects. Evaluated on the OpenEQA EM-EQA benchmark, using both global and local visual cues achieves an average LLM-Match score of $70.8$, improving the state of the art by $15.5$ percentage point. By reducing the number of frames processed, our approach increases correctness, reduces computational cost and improves explainability.

## Dataset

We use the [OpenEQA](https://github.com/facebookresearch/open-eqa/) dataset to evaluate our approach, specifically the [ScanNet](http://www.scan-net.org) subset. OpenEQA can be cloned via the link, to obtain the scannet dataset, please follow the instructions [here](https://github.com/ScanNet/ScanNet#scannet-data).

## Installation

The code requires a `python=3.9` environment, it has not been tested with other python versions. Please create an environment and use the bash script to setup the file structure and download the requirements.

```bash
bash ./setup.sh
```

Afterwards, please provide custom paths and huggingface token in the .yaml file.