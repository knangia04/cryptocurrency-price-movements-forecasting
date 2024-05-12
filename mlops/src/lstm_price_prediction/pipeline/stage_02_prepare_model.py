from lstm_price_prediction.config.configuration import ConfigurationManager
from lstm_price_prediction.components.prepare_model import PrepareModel
from lstm_price_prediction import logger



STAGE_NAME = "Prepare model"


class PrepareModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_model_config = config.get_prepare_model_config()
        prepare_model = PrepareModel(config=prepare_model_config)
        prepare_model.get_model()
        #prepare_model.update_base_model()



if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
