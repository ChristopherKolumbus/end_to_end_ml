import os
import requests
import shutil
import tarfile


def main():
    download_root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    housing_path = "datasets/housing"
    housing_url = download_root + housing_path + "/housing.tgz"
    fetch_housing_data(housing_url, housing_path)


def fetch_housing_data(housing_url, housing_path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    response = requests.get(housing_url, stream=True)
    response.raise_for_status()
    with open(tgz_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


if __name__ == '__main__':
    main()
