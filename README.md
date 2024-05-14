# group_10_project

## Team

## Overview

Our project aims to combine PyTorch, the [IEX Parser](https://gitlab.engr.illinois.edu/ie421_high_frequency_trading_spring_2024/iex-downloader-parser) developed in previous semesters, and the Illinois Campus Cluster to predict prices for cryptocurrency Spot ETFs that were recently approved on January 10th, 2024.

The IEX parser serves as the basis of our pipeline, providing a daily dump of data for selected symbols (BITO,FBTC,BITB,ARKB). This data is used for backtesting and predictions in a PyTorch LSTM model. Through running the project on the cluster, we hope to broaden access to ML training by eliminating computational and storage requirements.

## Usage

The project is split into two primary folders: a modified version of the IEX parser that runs on the cluster (./iex-campus-cluster), and the ML scripts (./rnn_model).

1) Upload all files onto the cluster using ccsetup.bash in the home directory.

2) Run the IEX Parser with sbatch ccparse.sbatch. Note you may need to change the "account" value for future semesters.

3) Train the LSTM model with sbatch rnn.sbatch.