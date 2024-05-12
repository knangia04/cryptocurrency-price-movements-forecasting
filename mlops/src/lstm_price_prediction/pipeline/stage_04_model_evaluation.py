from lstm_price_prediction.config.configuration import ConfigurationManager
from lstm_price_prediction.components.model_evaluation_mlflow import Evaluation
from lstm_price_prediction import logger
import os


STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/wko21/ie421_hft_spring_2024_group_10.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"]="wko21"
        os.environ["MLFLOW_TRACKING_PASSWORD"]="30a377a6d3c2bbba1e855843d402016843cbd34b"

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()





if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
            