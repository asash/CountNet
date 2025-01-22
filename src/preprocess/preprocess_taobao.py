# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import pandas as pd
from train_test_split import split_data
import zipfile

DATA_DIR = Path("__file__").absolute().parent / "data"
ARCHIVE_PATH = DATA_DIR / "raw/taobao/IJCAI16_data.zip"


def process_data(data: pd.DataFrame, dataset_name, n_val_users=1024):
    data = data[["user_id", "item_id", "timestamp"]]
    data = data.sort_values("timestamp")
    output_dir = Path(DATA_DIR/"processed"/dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "all.csv", index=False)
    split_data(dataset_name, n_val_users=n_val_users)


def main():
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Please download Taobao Dataset from https://tianchi.aliyun.com/dataset/53 and put it into the '{ARCHIVE_PATH.parent}' forlder")

    archive = zipfile.ZipFile(ARCHIVE_PATH)
    with  archive.open('ijcai2016_taobao.csv') as input:
        data = pd.read_csv(input)
    data = data.sort_values("time")
    data = data.rename(columns={"use_ID": "user_id", "ite_ID": "item_id", "time": "timestamp"})

    data_purchases = data[data['act_ID'] == 1]
    data_clicks = data[data['act_ID'] == 0]
    process_data(data_purchases, "taobao_purchases")
    process_data(data_clicks, "taobao_clicks")






if __name__ == "__main__":
    main()
