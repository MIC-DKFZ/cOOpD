# CRADL

> Contrastive Representations for unsupervised Anomaly Detection and Localization.
>
> Official Code Repository Going alongside the Paper
<!-- > Live demo [_here_](https://www.example.com). If you have the project hosted somewhere, include the link here. -->

## Table of Contents
- [Explain](#explain)
  - [Table of Contents](#table-of-contents)
  - [General Information](#general-information)
  - [Technologies Used](#technologies-used)
  - [Features](#features)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Project Status](#project-status)
  - [Room for Improvement](#room-for-improvement)
  - [Acknowledgements](#acknowledgements)


## General Information

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Python - version 3.7+
- See requirements.txt


## Features
Contrastive Pretext Training on Medical Data 



<!-- ## Screenshots
![Example screenshot](./img/screenshot.png) -->
<!-- If you have screenshots you'd like to share, include them here. -->


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

Verify that everythins is working
```
$ python train_pretext.py --fast_dev_run True
```
See in the logs folder, whether a log has been created,
access with tensorboard via:
```
$ tensorboard --path {path_to_logs}
```


## Usage
Anomaly Detection with the CRADL framework on 2D data.


## Project Status
Project is _in progress_
<!-- Project is: _in progress_ / _complete_ / _no longer being worked on_. If you are no longer working on it, provide reasons why. -->


## Room for Improvement
Anomaly Detection in 2D:
- Sample Level 
- Pixel Level 



## Acknowledgements
<!-- - This project was based on [this tutorial](https://www.example.com). -->


<!-- ## Contact -->
<!-- Created by [@flynerdpl](https://www.flynerd.pl/) - feel free to contact me! -->


<!-- Optional -->
## License 
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project