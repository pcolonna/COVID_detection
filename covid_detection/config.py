import os

root_data_dir = os.path.dirname(os.path.dirname(__file__)) + "/data/"

original_data_dir = {
    "normal": root_data_dir + "COVID-19 Radiography Database/NORMAL",
    "viral": root_data_dir + "COVID-19 Radiography Database/Viral Pneumonia",
    "covid": root_data_dir + "COVID-19 Radiography Database/COVID-19",
}

train_dirs = {
    "normal": root_data_dir + "train/normal",
    "viral": root_data_dir + "train/viral",
    "covid": root_data_dir + "train/covid",
}

test_dirs = {
    "normal": root_data_dir + "test/normal",
    "viral": root_data_dir + "test/viral",
    "covid": root_data_dir + "test/covid",
}
