from textSummerizer.config.configuration import ConfigurationManager
from textSummerizer.components.data_validation  import DataValidation
from textSummerizer.logging import logger

STAGE_NAME = "Data Validation Stage"

class DataValidationTrainingPipeline: 
    def __init__(self):
        pass 
    
    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_files_exist()
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>> {STAGE_NAME} completed >>>>>\n\nx=========x")
    except Exception as e:
        logger.exception(e)