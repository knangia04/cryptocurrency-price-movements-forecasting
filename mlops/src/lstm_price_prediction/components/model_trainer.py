import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from src.lstm_price_prediction.entity.config_entity import TrainingConfig,PrepareModelConfig
from pathlib import Path
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from src.lstm_price_prediction.components.rnn_model import rnn_models
import torch.nn as nn
from tqdm import tqdm
from src.lstm_price_prediction.constants import *
from src.lstm_price_prediction.utils.common import read_yaml
from src.lstm_price_prediction.components.rnn_model import preprocess, training_dataset 
import torch
import pandas as pd



class Training:
    def __init__(self, config: TrainingConfig, model_config: PrepareModelConfig):
        self.config = config
        self.model_config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params_filepath = PARAMS_FILE_PATH
        self.params = read_yaml(params_filepath)
        
    
    def get_base_model(self):
        self.model = rnn_models.LSTMModel(self.model_config.params_input_size,self.model_config.params_hidden_size ,self.model_config.params_num_layers, self.model_config.params_fc_layers_size).to(self.device)
        self.model.load_state_dict(torch.load(self.model_config.model_path))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config.params_learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def train_valid_generator(self):
        
        df = pd.read_csv(self.params.TRAIN_DATA)
        print('Preprocessing data....')
        new_df = preprocess.process_data(df)
        dataset = training_dataset.OrderBookDataset(new_df, self.params.SEQUENCE_LENGTH, self.params.PREDICT_EVENTS)
        print('Data preprocessing complete')
        train_loader, val_loader, test_loader = training_dataset.get_data_loaders(dataset, self.params.TEST_SPLIT, self.params.BATCH_SIZE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader


    
    @staticmethod
    def save_model(path: Path, model: torch.nn.Module):
        torch.save(model.state_dict(), path)



    
    def train(self):
        # Train the model in one epoch
        self.model.train()
        train_loss = 0
        for i, (x, y) in enumerate(tqdm(self.train_loader, desc="Training")):
            x = x.float()  # Convert input data to torch.float32 type
            y = y.long()  # Convert target data to torch.float32 type
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            y_hat = self.model(x)
            y_hat = y_hat.to(torch.float32)
            loss = self.criterion(y_hat, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(self.train_loader)

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

