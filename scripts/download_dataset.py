import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


if __name__ == "__main__":

    outputTrainCsv              = "./data/train.csv"
    outputUnicodeTranslation    = "./data/unicode_translation.csv"
    outputTrainImages           = "./data/train_images.zip"
    outputTestImages            = "./data/test_images.zip"
    outputs = [outputTrainCsv, outputUnicodeTranslation, outputTrainImages, outputTestImages]

    googleFileID_trainCSV                   = "1QkUqqFG3mOVvyvoOcOP3uaYmfgrLTxcU"
    googleFileID_outputUnicodeTranslation   = "1DIHqCuIgDcJkrik4YT1GagHzZ_rMtBPc"
    googleFileID_trainImages                = "1lyYZh5LwQUJCxCZ2EEqXPkQ5r1Gq3LzJ"
    googleFileID_testImages                 = "1Y8JgczT1chZY2DxLbMD2VF5emTbAYRCS"
    googleFilesID = [googleFileID_trainCSV, googleFileID_outputUnicodeTranslation
                     , googleFileID_trainImages, googleFileID_testImages]

    for out, fileID in zip(outputs, googleFilesID):
        download_file_from_google_drive(fileID, out)
    print('Download Complete!')

