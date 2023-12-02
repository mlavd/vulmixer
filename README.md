# Vul-Mixer

The official reproduction repository for "Vul-Mixer: Efficient and Effective Machine Learning–Assisted Software Vulnerability Detection".

![Repo Logo](logo.jpg)

<!-- Table of contents -->
<details open="open">
  <summary>Table of Contents</summary>

1. [How to Use](#how-to-use)
1. [Installation](#installation)
    1. [Environments](#environments)
1. [Datasets](#datasets)
    1. [Dataset Staging](#dataset-staging)
    1. [Function Formatting](#function-formatting)
1. [Model Training & Testing](#model-training--testing)
    1. [Caching Embeddings](#caching-embeddings)
    1. [Training](#training)
    1. [Model Weights](#model-weights)
    1. [Testing](#testing)
1. [Research Questions](#research-questions)
1. [Contributing / Updates](#contributing--updates)
1. [Acknowledgements](#acknowledgements)

</details>

## How to Use
This repository contains all the code and files necessary to reproduce the results from the "Vul-Mixer: Efficient and Effective Machine Learning–Assisted Software Vulnerability Detection" paper. If you encounter any issues with this repository, feel free to email the corresponding author for assistance.

## Installation
### Environments
Conda environments are provided in the `envs` directory. All resource-efficient models us the same environment named `fire`. The table below lists each environment's file, name, and use.

| File        | Name    | Use
|-------------|---------|----------
| `fire.yml`  | `fire`  | Framework for investigating resource-efficient MLAVD
| `clang.yml` | `clang` | Formatting datasets with clang-formatter

#### Using Environments
These commands create all environments.

```bash
conda env create -f envs/fire.yml
```

To activate an environment, such as `fire`:
```bash
conda activate fire
```

## Datasets
Our paper makes use of several datasets. Due to size restrictions, we do not provide them in this repository. However, scripts are available to perform the necessary preprocessing. Further, we maintain a copy of the datasets that we would be happy to share directly with you. Simply email the corresponding author.

### Dataset Staging
#### CodeXGLUE
1. Download the dataset using the instructions and scripts in the [CodeXGLUE repository](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection). You should have 3 files:
    * `test.jsonl`
    * `train.jsonl`
    * `valid.jsonl`
2. Move these files into `data/jsonl/codexglue`

#### D2A
1. Download the D2A Leaderboard Dataset (V1.0) from the [IBM Data Exchange](https://developer.ibm.com/exchanges/data/all/d2a/).
2. Place these three files into `data/inputs/d2a`:
    * `d2a_lbv1_function_dev.csv`
    * `d2a_lbv1_function_test.csv`
    * `d2a_lbv1_function_train.csv`
3. Run `python scripts/d2a_staging.py`. The following files should now be in `data/jsonl/d2a`:
    * `dev.jsonl`
    * `test.jsonl`
    * `train.jsonl`
4. Rename `dev.jsonl` to `valid.jsonl`.


#### Draper VDISC
1. Download the Draper VDISC dataset from [OSF.io](https://osf.io/d45bw/)
2. Place the three files into `data/inputs/draper`:
    * `VDISC_test.hdf5`
    * `VDISC_train.hdf5`
    * `VDISC_validate.hdf5`
3. Run `python scripts/draper_staging.py`. The following files should now be in `data/jsonl/draper`:
    * `test.jsonl`
    * `train.jsonl`
    * `validate.jsonl`
4. Rename `validate.jsonl` to `valid.jsonl`.

#### Big-Vul
1. Download the Big-Vul dataset from the [Big-Vul Repository](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset).
2. Place the following file into `data/inputs`:
    * `MSR_data_cleaned.csv`
3. Run `python scripts/bigvul_staging.py --vul=0`
4. Run `python scripts/bigvul_staging.py --vul=1`
5. Verify the follow files are in `data/jsonl/bigvul`.
    * `safe_after.jsonl`
    * `safe_before.jsonl`
    * `vuln_after.jsonl`
    * `vuln_before.jsonl`


#### DiverseVul
1. Download DiverseVul from the [DiverseVul Repository](https://github.com/wagner-group/diversevul)
2. Place the following file into `data/inputs`:
    * `diversevul_20230702.jsonl`
3. Run `python scripts/diversevul_staging.py`. The following files should now be in `data/jsonl/diversevul`:
    * `all.jsonl`
    * `test.jsonl`
    * `train.jsonl`
    * `valid.jsonl`

#### Wild C Subset
The Wild C dataset is available here: [https://github.com/mlavd/wild-c](https://github.com/mlavd/wild-c). It is quite large and can take time to download. Because the dataset is only used for profiling the training and inference time, the exact subset used should not affect the results significantly. We will be happy to provide the dataset on request.


### Function Formatting
To format the datasets according using Clang formatter, run the `scripts/format_dataset.py` script on each JSONL file to be formatted. For instance:

```bash
python scripts/format_dataset.py --input=data/jsonl/d2a/train.jsonl
```

Using the provided `clang` environment is the easiest way to install clang-format and all the necessary libraries.

## Model Training & Testing
### Caching Embeddings
Before training or testing the models, the embeddings must be cached. All models in the paper use `microsoft/graphcodebert-base` embeddings, but this can be customized. Use the following script to cache embeddings. Modifications may need to be made to the script to extract the appropriate matrix from the model.

```bash
python scripts/cache_embeddings.py --checkpoint=microsoft/graphcodebert-base
```

### Training
All the models can be trained using the same training script: `run.py`. This file pulls configuration files from the `config/data` and `config/model` directories. The following models and dataset configurations are provided with the repository:

| Datasets    | Models     |
|-------------|------------|
| `codexglue` | `avg-pool` |
| `d2a`       | `mha`      |
| `diverse`   | `mlp`      |
| `draper`    | `shift`    |
| `profile`   | `tca`      |
|             | `vulmixer` |

The configuration files may be modified to adjust the number of mixers and sizes. It's even possible to change between mixers within the same model. We encourage experimentation!

Here is an example of how to train Vul-Mixer on the D2A dataset.

```bash
python run.py --do_train --data=d2a --model=vulmixer
```

After training, metrics are logged for MLFlow in `logs/mlflow`. The path to the top checkpoint will be printed at the end of the file and may be moved to an different location for storage and testing.

The following configuration options are available as arugments to the `run.py` script.

| Argument      | Default         | Description                           |
|---------------|-----------------|---------------------------------------|
| `suffix`      | `''`            | Suffix for MLflow run name            |
| `checkpoint`  | `None`          | Weights to be loaded                  |
| `device`      | `cuda`          | Device to run the model on            |
| `max_epochs`  | `20`            | Number of epochs                      |
| `num_workers` | `8`             | Number of workers for dataset loading |
| `log`         | `./logs/mlflow` | Path for MLFlow logs                  |

### Model Weights
Due to the size of model weights, we do not provide them in this repository. We would be happy to share them directly; simply email the corresponding author.

### Testing
Testing may be performed with the same script as training.

```bash
python run.py --do_test --data=d2a --model=vulmixer --checkpoint=[path to weights]
```

## Research Questions
The following sections describe the steps required to reproduce the results from each of the research questions.

### RQ1: How efficient is Vul-Mixer compared to the baselines?
1. Use the scripts provided at [mlavd/model_gaps](https://github.com/mlavd/model_gaps) to train and test the baseline models against the Wild C profile dataset. Time the execution.
2. Use the `run_profile.py` script to time each resource-efficient baseline:
    * `python run_profile.py --model=vulmixer --profile=test`
    * `python run_profile.py --model=vulmixer --profile=train`

### RQ2: How effective is Vul-Mixer compared to resource-efficient baselines?
1. Use the above instructions to train and test the resource-efficient baselines against all the datasets.
2. Use the above instructions to train and test Vul-Mixer against all the datasets.
3. Use the equations provided in the paper to calculate MAP and GAP.

### RQ3: How effective is Vul-Mixer compared to MLAVD baselines?
1. Use the scripts and weights provided at [mlavd/model_gaps](https://github.com/mlavd/model_gaps) to test MLAVD baselines against CodeXGLUE, D2A, and Draper VDISC.
2. Use the above instructions to train and test Vul-Mixer against all the datasets.
3. Use the equations provided in the paper to calculate MAP and GAP.

### RQ4: How cost-effective is Vul-Mixer compared to baselines?
1. Use the results from the above research questions and the equations provided in the paper to calculate the cost-effectiveness ratios.

### RQ5: Is Vul-Mixer useful in real-world settings?
1. Use the scripts provided at [mlavd/model_gaps](https://github.com/mlavd/model_gaps) to test the MLAVD baselines against the DiverseVul and Big-Vul. Name the prediction outputs with the format: `[model]_[trainin_dataset]_on_[test dataset].txt`. For instance: `codebert_codexglue_on_big_safe_after.txt`. Place these results in `logs/predictions`.
2. Use `run_diversevul.py` to run Vul-Mixer on DiverseVul. For each run, a CSV file is produced in the root directory. Move this file to `logs/diversevul/vulmixer/[training_dataset].csv`.
3. Use `run_bigvul.py` to run Vul-Mixer on Big-Vul. For each run, a CSV file is produced in the root directory. Move this file to `logs/bigvul/vulmixer/[traininin_dataset]-[safe/vuln].csv`.
4. Run `scripts/join_bigvul.py` and `convert_diversevul.py` to combine the predictions.
5. Run `notebooks/bigvul.ipynb` and `notebooks/diversevul.ipynb` to produce the final results.


## Contributing / Updates
As a reproduction repository, no updates will be made to this repository unless they are required to fix a bug in the code. Updates will only be made by the original authors. A log of updates will be provided here.

### Updates
- No updates have been made

## Acknowledgements
Thank you to the authors of the original models, whose reproduction repositories we referenced and used as the basis of our code:

- CodeBERT from [microsoft/CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection/code)
- CoTeXT from [Original Authors via Huggingface](https://huggingface.co/razent/cotext-1-ccg)
- LineVul from [awsm-research/LineVul](https://github.com/awsm-research/LineVul)
- ReGVD from [daiquocnguyen/GNN-ReGVD](https://github.com/daiquocnguyen/GNN-ReGVD)
- TCAMixer from [Liu-Xiaoyan97/TCAMixer](https://github.com/Liu-Xiaoyan97/TCAMixer)
