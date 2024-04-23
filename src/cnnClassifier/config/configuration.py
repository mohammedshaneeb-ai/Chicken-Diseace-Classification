from cnnClassifier.constants import CONFIG_FILE_PATH,PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml,create_directories
from cnnClassifier.entity import DataIngestionConfig,PrepareBaseModelConfig


class ConfigurationManager:
    def __init__(self,
        config_file_path = CONFIG_FILE_PATH,
        params_file_path = PARAMS_FILE_PATH
        ):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig (
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params
        create_directories([config.root_dir])
        
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir = config.root_dir,
            base_model_path = config.base_model_path,
            updated_base_model_path= config.updated_base_model_path,
            params_image_size= params.IMAGE_SIZE,
            params_learning_rate= params.LEARNING_RATE,
            params_include_top= params.INCLUDE_TOP,
            params_weights= params.WEIGHTS,
            params_classes= params.CLASSES
        )
        return prepare_base_model_config
    