import os
from src.lstm_price_prediction.constants import *
from src.lstm_price_prediction.utils.common import read_yaml, create_directories
from src.lstm_price_prediction.entity.config_entity import (DataIngestionConfig,
                                                PrepareModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_prepare_model_config(self) -> PrepareModelConfig:
        config = self.config.prepare_model
        
        create_directories([config.root_dir])

        prepare_model_config = PrepareModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            params_input_size = self.params.INPUT_SIZE,
            params_hidden_size = self.params.HIDDEN_SIZE,
            params_num_layers = self.params.NUM_LAYERS,
            params_fc_layers_size = self.params.FC_LAYERS_SIZE,
            params_learning_rate = self.params.LEARNING_RATE
   
        )

        return prepare_model_config
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        params = self.params
        untrained_model = self.config.prepare_model
        training_data_path = "..data/20220801_book_updates.csv"
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            untrained_model_path=Path(untrained_model.model_path),
            trained_model_path=Path(training.trained_model_path),
            mlflow_uri = "https://dagshub.com/wko21/ie421_hft_spring_2024_group_10.mlflow",
            training_data=training_data_path,
            all_params=self.params,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_sequence_length=params.SEQUENCE_LENGTH,
            params_predict_events=params.PREDICT_EVENTS,
            params_val_interval=params.VAL_INTERVAL
        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=self.config.training.trained_model_path,
            training_data="20220801_book_updates.csv",
            mlflow_uri="https://dagshub.com/wko21/ie421_hft_spring_2024_group_10.mlflow",
            all_params=self.params,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config


      