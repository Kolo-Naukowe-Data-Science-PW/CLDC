from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
import os.path

def downloaded(path='Data_zip/'):
    "Check if the ZIP file is already downloaded."
    return os.path.exists(os.path.join(path, 'cassava-leaf-disease-classification.zip'))


def extracted(path='data/'):
    "Check if the ZIP file is already extracted."

    api = KaggleApi()
    api.authenticate()

    for file in api.competition_list_files('cassava-leaf-disease-classification'):
        if not (os.path.exists(os.path.join(path, str(file)))):
            return False

    return True


def load_data_from_kaggle(data_path='data/', zip_path='Data_zip/'):
    "Import data from kaggle, unzip it, and store it in data folder"

    api = KaggleApi()
    api.authenticate()

    if not downloaded(zip_path):
        print("The ZIP file is being downloaded...")
        api.competition_download_files('cassava-leaf-disease-classification', path=zip_path)
        print("Done.\n")

    if not extracted(data_path):
        print("The ZIP file is being extracted...")
        zf = ZipFile(os.path.join(zip_path, 'cassava-leaf-disease-classification.zip'))
        zf.extractall(path=data_path)
        print("Done.\n")