from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def load_housing_data():
    data_file_path = Path("datasets/housing.tgz")
    if not data_file_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        data_url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
        urllib.request.urlretrieve(data_url, data_file_path)
        with tarfile.open(data_file_path) as housing_tgz:
            housing_tgz.extractall(path="datasets")
    return pd.read_csv("datasets/housing.csv")


def shuffle_and_split_data(data, test_ratio):
    indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == '__main__':
    housing_data = load_housing_data()

    print("Dataset Informations\n")
    print(housing_data.info())

    print("\n**************\n")

    print("Dataset Head\n")
    print(housing_data.head())

    print("\n**************\n")

    print("Ocean Proximity Value Counts\n")
    print(housing_data["ocean_proximity"].value_counts())

    print("\n**************\n")

    print("Dataset Description\n")
    print(housing_data.describe())

    print("\n**************\n")

    print("Dataset Histogram\n")
    housing_data.hist(bins=50, figsize=(20, 15))
    plt.show()





