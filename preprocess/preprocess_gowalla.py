from pathlib import Path
import pandas as pd
from train_test_split import split_data
import gzip


DATA_DIR = Path("__file__").absolute().parent / "data"

ARCHIVE_PATH = DATA_DIR / "raw/gowalla/loc-gowalla_totalCheckins.txt.gz"



def process_data(data: pd.DataFrame, dataset_name, n_val_users=1024):
    data = data[["user_id", "item_id", "timestamp"]]
    data = data.sort_values("timestamp")
    output_dir = Path(DATA_DIR/"processed"/dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "all.csv", index=False)
    split_data(dataset_name, n_val_users=n_val_users)


def main():
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Please download Gowalla Dataset from https://snap.stanford.edu/data/loc-gowalla.html and put it into the {ARCHIVE_PATH.parent} folder")

    with gzip.open(ARCHIVE_PATH) as input:
        data = pd.read_csv(input, delimiter="\t", names="user_id,timestamp,lat,lon,item_id".split(","))
    data.timestamp = pd.to_datetime(data['timestamp']).apply(lambda x: x.timestamp()).astype('int64')
    process_data(data, "gowalla")






if __name__ == "__main__":
    main()
