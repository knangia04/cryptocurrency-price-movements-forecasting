import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from urllib.parse import urlparse
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
from src.lstm_price_prediction.utils.common import read_yaml, classification_metrics
from src.lstm_price_prediction.components.rnn_model import preprocess, training_dataset 
import torch
import pandas as pd
import mlflow


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

    def validate(self):
        self.load_model(self.model_path)
        self.model.eval()
        all_y_true = torch.LongTensor()
        all_y_pred = torch.LongTensor()
        val_loss = 0
        for x, y in self.val_loader:
            x = x.float()  # Convert input data to torch.float32 type
            y = y.long()  # Convert target data to torch.float32 type
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            val_loss += loss.item()
            y_hat = F.softmax(y_hat, dim=1)
            y_pred = torch.argmax(y_hat, dim=1)
            all_y_true = torch.cat((all_y_true, y.to('cpu').long()), dim=0)
            all_y_pred = torch.cat((all_y_pred,  y_pred.to('cpu').long()), dim=0)
        val_loss = val_loss / len(self.test_loader)
        acc,  precision, recall, f1 = classification_metrics(all_y_pred.detach().numpy(), 
                                                                all_y_true.detach().numpy())
        return val_loss, acc, precision, recall, f1
    
    @staticmethod
    def save_model(path: Path, model: torch.nn.Module):
        torch.save(model.state_dict(), path)



    
    def train(self):
        # Train the model in one epoch
        self.model.train()
        train_loss = 0
        for epoch in range(self.params.EPOCHS): 
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
            if epoch % 10 == 0:
                val_loss, acc, precision, recall, f1 = self.validate(self.model, self.val_loader, weight = None)
                print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} Validation Loss: {val_loss:.6f} acc: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}")
                self.log_into_mlflow()
    
    def log_into_mlflow(self, train_loss, val_loss, val_acc, val_f1score):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"training loss":train_loss, "validation loss": val_loss, "accuracy": val_acc, "f1score": val_f1score}
            )
                