import requests
import os
from pathlib import Path
import platform
import zipfile

def download_file_from_google_drive(gdriveID, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': gdriveID}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gdriveID, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 1024

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


if __name__ == "__main__":
    outputTrainCsv = "./data/train.csv"
    outputUnicodeTranslation = "./data/unicode_translation.csv"
    outputTrainImages = "./data/train.zip"
    outputTestImages = "./data/test.zip"
    outputTrainChar  = "./data/train_char.zip"
    outputs = [outputTrainCsv, outputUnicodeTranslation, outputTrainImages, outputTestImages, outputTrainChar]

    googleFileID_trainCSV = "1QkUqqFG3mOVvyvoOcOP3uaYmfgrLTxcU"
    googleFileID_outputUnicodeTranslation = "1DIHqCuIgDcJkrik4YT1GagHzZ_rMtBPc"
    googleFileID_trainImages = "1lyYZh5LwQUJCxCZ2EEqXPkQ5r1Gq3LzJ"
    googleFileID_testImages = "1Y8JgczT1chZY2DxLbMD2VF5emTbAYRCS"
    googleFileID_trainChar = "1yNpdCgTMdDoBKId5ksj2O7EgFverEKDC"
    googleFilesID = [googleFileID_trainCSV, googleFileID_outputUnicodeTranslation, googleFileID_trainImages,
                     googleFileID_testImages, googleFileID_trainChar]

    for out, fileID in zip(outputs, googleFilesID):
        download_file_from_google_drive(fileID, out)
    print('Download Complete!')
    print('Uncompressing files ...')
    dataDir = os.path.join(Path(__file__).parent.parent, 'data')

    os_name = platform.system()
    if os_name == 'Linux':
        os.system(f"unzip \"{os.path.join(dataDir, 'train.zip')}\" -d \"{os.path.join(dataDir, 'train')}\"")
        os.system(f"unzip \"{os.path.join(dataDir, 'test.zip')}\" -d \"{os.path.join(dataDir, 'test')}\"")
        os.system(f"unzip \"{os.path.join(dataDir, 'train_char.zip')}\" -d \"{os.path.join(dataDir, 'train_char')}\"")
        os.system(f"rm \"{os.path.join(dataDir, 'train_.zip')}\"")
        os.system(f"rm \"{os.path.join(dataDir, 'test.zip')}\"")
        os.system(f"rm \"{os.path.join(dataDir, 'train_char.zip')}\"")
    elif os_name == 'Windows':
        with zipfile.ZipFile(outputTrainImages, 'r') as zip_ref:
            zip_ref.extractall("./data/train")
        with zipfile.ZipFile(outputTestImages, 'r') as zip_ref:
            zip_ref.extractall("./data/test")
        with zipfile.ZipFile(outputTrainChar, 'r') as zip_ref:
            zip_ref.extractall("./data/train_char")
        os.system(f"del {os.path.join(dataDir, 'train_char.zip')}")
        os.system(f"del {os.path.join(dataDir, 'train.zip')}")
        os.system(f"del {os.path.join(dataDir, 'test.zip')}")

    print('All done! :D')
