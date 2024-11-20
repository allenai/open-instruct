# Synthetic Preference Pipeline

This directory contains the implementation of the synthetic data pipeline for Tulu 3.
This pipeline is based on the Ultrafeedback pipeline ([Cui et al., 2023](https://arxiv.org/abs/2310.01377)) but with modifications such as the inclusion of on-policy data during data generation, and the use of GPT-4o for preference annotation.

Here's an overview of the pipeline (and how each script corresponds to each component):

![](scripts/synth_pref/assets/ufpp_pipeline_v2_normal.png)
![](scripts/synth_pref/assets/ufpp_pipeline_v2_code.png)

## Setup

You need to install specific dependencies for this pipeline:

```sh
python3 -m venv venv
pip install -r scripts/synth_pref/requirements.txt
```

## How-to-use
