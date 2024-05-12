from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class PrepareModelConfig:
    root_dir: Path
    model_path: Path
    params_input_size: int
    params_hidden_size: int
    params_num_layers: int
    params_fc_layers_size: int
    params_learning_rate: float
   


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    untrained_model_path: Path
    mlflow_uri: str
    training_data: Path
    all_params: dict
    params_epochs: int
    params_batch_size: int
    params_predict_events: int
    params_sequence_length: int
    params_val_interval: int


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_batch_size: int
    
