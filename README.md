# group_10_project

## Team

## Overview

The volatile and dynamic nature of cryptocurrency markets presents both opportunities and challenges for investors, traders, and enthusiasts alike. In this project, we aim to leverage the power of machine learning to develop predictive models capable of forecasting cryptocurrency price movements. By harnessing historical price data using previous semesters’ projects and the computing power provided by the university, we seek to build robust and accurate models that can provide valuable insights into the behavior of cryptocurrency markets. Through this endeavor, we aim to gain a deeper understanding of the underlying factors driving cryptocurrency price dynamics, the machine learning development process, and hands-on experience with cutting-edge computing technology.

Our project aims to combine PyTorch, the [IEX Parser](https://gitlab.engr.illinois.edu/ie421_high_frequency_trading_spring_2024/iex-downloader-parser) developed in previous semesters, and the Illinois Campus Cluster to predict prices for cryptocurrency Spot ETFs that were recently approved on January 10th, 2024. PyTorch serves as a robust framework for implementing machine learning algorithms, specifically Recurrent Neural Networks (RNN), which are well-suited for time-series forecasting tasks like predicting price movements. PyTorch's flexibility allows for easy experimentation with model architectures, hyperparameters, and loss functions, facilitating the exploration of various approaches to improve prediction accuracy.

The IEX parser serves as the basis of our pipeline, providing a daily dump of data for selected symbols (BITO,FBTC,BITB,ARKB). This data is used for backtesting and predictions in a PyTorch LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) models. Through running the project on the cluster, we hope to broaden access to ML training by eliminating computational and storage requirements. Finally, we priotitized the integration of MLOps (Machine Learning Operations) practices to ensure the efficiency, scalability, and reliability of our development cycle. Using MLFlow to organize the machine learning experiments into runs into experiment, each run captured the metadata such as parameters and metrics. We then used the MLFlow API to log the data related to training and evaluation.

## Milestones Accomplished
1) Got the data from the IEX parser. Developed basic visualizations and preprocess the data (remove outliers, normalization, missing values).  
2) Developed an LSTM prototype on a smaller portion of the IEX dataset, before expanding it to other RNN models like GRU and modularizing the machine leanring code.
3) Integrated the actual IEX parser data into the model prototype.
4) Integrated a backtesting script to the model (see backtest.py and incorporation in main.py under rnn_model directory).
5) Migrated the computing to the Campus Cluster.
5) Fine-tuned the model to account for over/underfitting comparing the results from different optimization techniques.
6) Incorporated MLOps into the model using MLFlow API.


## Usage

The project is split into two primary folders: a modified version of the IEX parser that runs on the cluster (./iex-campus-cluster), and the ML scripts (./rnn_model).

1) Upload all files onto the cluster using ccsetup.bash in the home directory.

2) Run the IEX Parser with sbatch ccparse.sbatch. Note you may need to change the "account" value for future semesters.

3) Train the LSTM/GRU model with sbatch rnn.sbatch.

## Team Members

Joseph Chen:

Connor Flynn: UIUC Computer Science & Linguistics (Spring 2025), cjflynn2@illinois.edu, [LinkedIn](https://www.linkedin.com/in/connor-flynn-253960228/) 

Krish Nangia: UIUC Computer Engineering (December 2025), knang2@illinois.edu, [Linkedin](https://www.linkedin.com/in/krish-nangia-uiuc)

My technical toolkit includes proficiency in programming languages such as Java, Python, C, C++, and more. I have explored the world of machine learning with frameworks like TensorFlow, PyTorch, and Scikit-Learn, and also have harnessed the power of data science libraries like Pandas, Matplotlib, and Numpy. Additionally, I have knowledge in SQL and web frameworks like Flask. 

​I am very passionate about leveraging technology for the betterment of today's society. I hope to explore different aspects of software development and data science/analytics to ultimately use the skills gained to help further advance technology in other fields.

Ray Ko:
