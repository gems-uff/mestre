import configs
import os
from google_drive_downloader import GoogleDriveDownloader as gdd


def main():
    print('Beginning file download')

    zip_file_path = f'{configs.DATA_PATH}/data.zip'
    # if not os.path.exists(f'{configs.DATA_PATH}'):
    #     os.mkdir(f'{configs.DATA_PATH}')
    gdd.download_file_from_google_drive(file_id='1OwMq81W2xajsHuG6HKBzZoL0Fp3_HL5d',
                                    dest_path='../data.zip',
                                    unzip=True, overwrite=True)
    if os.path.exists('../data.zip'):
        os.remove('../data.zip')
    print('Finished')
main()