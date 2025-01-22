# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

def project_root():
    current_dir = Path(__file__).absolute().parent.parent.parent
    return current_dir

def data_root():
    res = project_root()/"data"
    res.mkdir(parents=True, exist_ok=True)
    return res

def raw_data_root():
    res = data_root() / "raw"
    res.mkdir(parents=True, exist_ok=True)
    return res

def processed_data_root() -> Path:
    res = data_root() / "processed"
    res.mkdir(parents=True, exist_ok=True)
    return res

def dataset_splits_dir(dataset):
    res: Path = processed_data_root() / dataset / "split"
    res.mkdir(exist_ok=True, parents=True)
    return res
 

def output_root():
    res =  project_root() / "output" 
    res.mkdir(parents=True, exist_ok=True)
    return res
    

def get_checkpoints_root():
    res =  output_root() / "checkpoints"
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_tensorboard_root():
    res =  output_root() / "tensorboard"
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_model_dataset_tensorboard_dir(dataset, model):
    res = get_tensorboard_root() / dataset / model
    res.mkdir(parents=True, exist_ok=True)
    return res




def get_dataset_checkpoints_dir(dataset):
    res = get_checkpoints_root() / dataset
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_recs_dir(dataset):
    res = output_root() / "recs" / dataset
    res.mkdir(parents=True, exist_ok=True)
    return res    

def get_src_dir():
    return project_root() / "src" 

def get_analysis_dir():
    return get_src_dir() / "analysis" 

def get_training_scripts_dir():
    return get_src_dir() / "training"

def get_eval_scripts_dir():
    return get_src_dir() / "eval"




if __name__ == "__main__":
    print(get_checkpoints_root())

