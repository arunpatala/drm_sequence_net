import requests

def download_from_s3(bucket_name, file_key, local_path):
    """
    Download a publicly available file from S3.

    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - file_key (str): The key (path) of the file in the S3 bucket.
    - local_path (str): The local path where you want to save the file.

    Returns:
    - bool: True if the file was downloaded successfully, False otherwise.
    """
    file_url = f"https://{bucket_name}.s3.amazonaws.com/{file_key}"

    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        with open(local_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        return True
    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

# Example Usage:

import os

def download():
    if not os.path.exists("DATA"):
        os.mkdir("DATA")
    if not os.path.exists("DATA/data.zip"):
        print("downloading data.zip .....")    
        download_from_s3("drmsequencenet", "data.zip", "DATA/data.zip")
        print(" downloaded.")    
    if not os.path.exists("DATA/EXPS/BEST/model_epoch_best.th"):            
        if not os.path.exists("DATA/EXPS/BEST"):
            os.makedirs("DATA/EXPS/BEST")
        print("downloading model ...")
        download_from_s3("drmsequencenet", "model_epoch_best.th", "DATA/EXPS/BEST/model_epoch_best.th")
        print(" downloaded.")
    

import zipfile
import os

def unzip_to_folder(zip_file_path, output_folder='DATA'):
    """
    Unzips the specified zip file to the specified output folder.

    Parameters:
    - zip_file_path (str): Path to the zip file to be extracted.
    - output_folder (str): Path to the folder where the zip file should be extracted.

    Returns:
    - None
    """
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)

    print(f"'{zip_file_path}' has been extracted to '{output_folder}'")

def download_and_unzip():
    download()
    if not os.path.exists("DATA/data"):
        unzip_to_folder('DATA/data.zip')

if __name__ == "__main__":
    download_and_unzip()

# Example usage:
#unzip_to_folder('data.zip', 'DATA')
