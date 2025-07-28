import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

# Initialize the logger
logger = get_logger(__name__)

# Crete data ingestion class
class DataIngestion:
    def __init__(self, config):
        self.config = config['data_ingestion']
        self.bucket_name = self.config['bucket_name']
        self.file_name = self.config['bucket_file_names']

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info('🚀 Initiating the data ingestion process: Preparing to fetch raw data from GCP bucket...')

    
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            for file_name in self.file_name:
                file_path = os.path.join(RAW_DIR, file_name)

                if file_name == "animelist.csv":
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)

                    data = pd.read_csv(file_path, nrows=5000000) # Limit the number of raws for fast training this has 70 million rows
                    data.to_csv(file_path, index=False)

                    logger.info(f'🎬 Raw data "{file_name}" secured and optimized — 5 million rows ready for action. Location: {file_path}')
                
                else:
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)

                    logger.info(f'📦 File "{file_name}" fetched from bucket and saved locally at: {file_path}')
        
        except Exception as e:
            logger.error(f'❌ Error while downloading data from GCP bucket "{self.bucket_name}": {str(e)}')
            raise CustomException('Failed to download data', e)
    

    def run(self):
        try:
            logger.info('🚀 Starting the data ingestion pipeline...')
            self.download_csv_from_gcp()
            logger.info('✅ Data ingestion completed successfully.')
        
        except CustomException as ce:
            logger.error(f'❌ CustomException occurred during data ingestion: {str(ce)}')
        
        finally:
            logger.info('📦 Data ingestion routine has exited.')


if __name__=="__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()