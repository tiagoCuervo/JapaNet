import requests


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
    outputTrainChar = "./data/train_char.zip"
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
