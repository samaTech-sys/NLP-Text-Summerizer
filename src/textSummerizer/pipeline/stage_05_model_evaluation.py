from textSummerizer.config.configuration import ConfigurationManager
from textSummerizer.components.model_evaluation  import ModelEvaluation
from textSummerizer.logging import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline: 
    def __init__(self):
        pass 
    
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config =config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.evaluate()
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>> {STAGE_NAME} completed >>>>>\n\nx=========x")
    except Exception as e:
        logger.exception(e)