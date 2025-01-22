# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import pandas as pd
from train_test_split import split_data


DATA_DIR = Path("__file__").absolute().parent / "data"

DATASET_PATHS = [DATA_DIR / Path("raw/booking/train_set.csv"), DATA_DIR / Path("raw/booking/test_set.csv")]


def process_data(data: pd.DataFrame, dataset_name, n_val_users=1024):
    data = data[["user_id", "item_id", "timestamp"]]
    data = data.sort_values("timestamp")
    output_dir = Path(DATA_DIR/"processed"/dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "all.csv", index=False)
    split_data(dataset_name, n_val_users=n_val_users)


def main():
    partitions = []
    for path in  DATASET_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"Please download Booking.com dataset from https://github.com/bookingcom/ml-dataset-mdt and put it into the '{path.parent}' forlder")
        with open(path) as input:
            data = pd.read_csv(input)
        pass
        data["timestamp"] = pd.to_datetime(data['checkin']).apply(lambda x: x.timestamp()).astype('int64')
        data["item_id"] = data["city_id"]
        data = data[["user_id", "item_id", "timestamp"]]
        partitions.append(data)
    full_data = pd.concat(partitions)
    process_data(full_data, "booking")

if __name__ == "__main__":
    main()
