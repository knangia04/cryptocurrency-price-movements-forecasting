from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from lstm_price_prediction.entity.config_entity import EvaluationConfig
from lstm_price_prediction.utils.common import read_yaml,save_json
from lstm_price_prediction.constants import *
from src.lstm_price_prediction.components.rnn_model import preprocess, training_dataset 
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import *
from src.lstm_price_prediction.components.rnn_model import rnn_models

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params_filepath = PARAMS_FILE_PATH
        self.params = read_yaml(params_filepath)
        self.model_path = self.config.path_of_model
        df = pd.read_csv(self.params.TRAIN_DATA)
        print('Preprocessing data....')
        new_df = preprocess.process_data(df)
        dataset = training_dataset.OrderBookDataset(new_df, self.params.SEQUENCE_LENGTH, self.params.PREDICT_EVENTS)
        print('Data preprocessing complete')
        train_loader, val_loader, test_loader = training_dataset.get_data_loaders(dataset, self.params.TEST_SPLIT, self.params.BATCH_SIZE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        

    def evaluation(self):
        self.load_model(self.model_path)
        self.model.eval()
        all_y_true = torch.LongTensor()
        all_y_pred = torch.LongTensor()
        val_loss = 0
        for x, y in self.test_loader:
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
        acc,  precision, recall, f1 = Evaluation.classification_metrics(all_y_pred.detach().numpy(), 
                                                                all_y_true.detach().numpy())
        self.score = [val_loss,acc,f1]
    
    @staticmethod
    def classification_metrics(Y_pred, Y_true):
        acc, precision, recall, f1score = accuracy_score(Y_true, Y_pred), \
                                            precision_score(Y_true, Y_pred, average='weighted', zero_division = 1), \
                                            recall_score(Y_true, Y_pred, average='weighted'), \
                                            f1_score(Y_true, Y_pred, average='weighted')
        return acc,  precision, recall, f1score
    
    def load_model(self, path: Path) -> torch.nn.Module:
        self.model = rnn_models.LSTMModel(self.params.INPUT_SIZE, self.params.HIDDEN_SIZE ,self.params.NUM_LAYERS, self.params.FC_LAYERS_SIZE).to(self.device)
        self.model.load_state_dict(torch.load(path))

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1], "f1score": self.score[2]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="LSTMModel")
            else:
                mlflow.pytorch.log_model(self.model, "model")
