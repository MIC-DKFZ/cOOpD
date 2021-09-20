# CRADL

> Contrastive Representations for unsupervised Anomaly Detection and Localization.
>
> Official Code Repository Going alongside the Paper


## Table of Contents
- [CRADL](#cradl)
  - [Table of Contents](#table-of-contents)
  - [General Information](#general-information)
  - [Technologies Used](#technologies-used)
  - [Features](#features)
  - [Setup](#setup)
    - [Running with new datasets](#running-with-new-datasets)
    - [Changing the generative models on the latent space](#changing-the-generative-models-on-the-latent-space)
  - [Usage](#usage)
  - [Project Status](#project-status)
  - [Room for Improvement](#room-for-improvement)
  - [Acknowledgements](#acknowledgements)
  - [License](#license)


## General Information



## Technologies Used
- Python - version 3.7+
- See requirements.txt
- [FrEIA Framework](https://github.com/VLL-HD/FrEIA)


## Features
Contrastive Pretext Training on Medical Data 



## Setup
Set this up with conda:
```
$ conda create -n cradl python=3.7
$ pip install -r requirements.txt
```

Set all the paths to the data and logs in: 
- config/global_config
- config/datasets/brain.py
- create own datamodule carrying your data 
  - see --> datamodules/brain_module.py \& atamodules/brain.py

Verify that everything is working
```
$ python train_pretext.py --fast_dev_run True
```
See in the logs folder, whether a log has been created,
access with tensorboard via:
```
$ tensorboard --path {path_to_logs}
```

Fitting of the generative models
```
$ python train_latent_gen.py -p {path_to_experiment}
```

Evaluation of the generative models
```
$ python eval.py -p {path_to_experiment}
```

### Running with new datasets 
To run experiments with new datasets:
1.  Add a new datamodule
2.  Add a new option for `get_data_loaders(name_exp)` in `eval.py`

### Changing the generative models on the latent space
To do this, add a new entry to the dictionary in the file `config/latent_model.py`.

## Usage
Anomaly Detection with the CRADL framework on 2D data.


## Project Status
Project is _in progress_



## Room for Improvement
Anomaly Detection in 2D:
- Sample Level 
- Pixel Level 



## Acknowledgements

## License 
