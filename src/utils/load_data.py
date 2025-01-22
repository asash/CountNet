# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import pandas as pd
import numpy as np

def set_str_dtype(df):
    df.user_id = df.user_id.astype("str")
    df.item_id = df.item_id.astype("str")
    return df


def load_data(dataset):
    dataset_dir = Path(__file__).absolute().parent.parent.parent / "data/processed" / dataset / "split"
    result = {}
    for file in dataset_dir.iterdir():
        partition, size = file.stem.split(".")
        size = int(size)
        if  partition in ["train", "val_all"]:
            data = np.memmap(file, dtype="int32", mode="r", shape=(size, 3))
            df = pd.DataFrame(data, columns=["user_id", "item_id", "timestamp"])
            result[partition] = set_str_dtype(df)
        elif partition in ["val", "test"]:
            data = np.memmap(file, dtype="int32", mode="r", shape=(size, 4))
            df = pd.DataFrame(data, columns=["user_id", "item_id", "timestamp", "is_repetition"])
            result[partition] = set_str_dtype(df) 
    return result
