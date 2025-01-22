# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
import json
from src.recommenders.SASRec import SASRecRecommender, SASRecConfig
import dill

from src.utils.load_data import load_data
from src.sagemaker.sagemaker_utils import sagemaker_tb_dir, is_running_on_sagemaker, sagemaker_model_output_dir, sagemaker_data_output_dir
from src.utils.project_dirs import get_dataset_checkpoints_dir, get_model_dataset_tensorboard_dir, get_recs_dir, get_src_dir
from src.utils.get_training_params import get_training_parameters
from src.recommenders.utils.logger import TrainingLogger
from multiprocessing import Process
from subprocess import check_call

from src.utils.validate_config import get_config_dict

parameters = get_training_parameters((("dataset", str, "booking"),
                                      ("device", str, "cuda:0"),
                                      ("max-batches-per-epoch", int, 128),
                                      ("max-epochs", int, 10000),
                                      ("batch-size", int, 512),
                                      ("effective-batch-size", int, 512),
                                      ("dropout", float, 0.5),
                                      ("embedding-size", int, 256),
                                      ("num-layers", int, 3),
                                      ("dim-feedforward", int, 1024),
                                      ("nhead", int, 4)
                                      ))


eval_script = get_src_dir() / "eval" / "eval_checkpoints.py"


if parameters["job_name"] is not None:
    job_id = parameters["job_name"]
else:
    job_id = datetime.now().strftime('%Y%m%d%H%M%S')

dataset = load_data(dataset = parameters['dataset'])
model_name = f"SASRec_{parameters['dataset']}_{job_id}"

if is_running_on_sagemaker():
    tensorboard_dir = sagemaker_tb_dir()
    output_dir = sagemaker_model_output_dir()
    data_output_dir = sagemaker_data_output_dir()

else:
    tensorboard_dir = get_model_dataset_tensorboard_dir(parameters['dataset'], model_name)
    output_dir = get_dataset_checkpoints_dir(parameters['dataset'])
    data_output_dir = get_recs_dir(parameters["dataset"])

print("Model Training parameters:", parameters)
print("Tensorboard dir:", tensorboard_dir)
print("Output dir:", output_dir)

out_file =  output_dir/ f"{model_name}.dill"

config = SASRecConfig(device=parameters['device'], batches_per_epoch=parameters['max_batches_per_epoch'],
                           max_epoch=parameters["max_epochs"],
                           batch_size =parameters["batch_size"],
                           effective_batch_size =parameters["effective_batch_size"],
                           dropout=parameters["dropout"],
                           embedding_size=parameters["embedding_size"],
                           num_layers=parameters["num_layers"],
                           dim_feedforward=parameters["dim_feedforward"],
                           nhead = parameters["nhead"]
                    )
hp_dict = get_config_dict(config)


def train():
    recommender = SASRecRecommender(config)
    recommender.train(dataset["train"], dataset["val"], dataset["val_all"], tensorboard_dir=tensorboard_dir)
    with open(out_file, "wb") as out:
        print(f"saving model checkpoint to {out_file}")
        dill.dump(recommender, out)

#  Training in a sandbox process so that we are sure that the resources (e.g. cuda memory) are cleaned up before evaluation.
training_process = Process(target=train)
training_process.start()
training_process.join()


#  evaluation process
check_call(["python3", str(eval_script.absolute()), "--absolute-path", "True", "--dataset", parameters['dataset'], "--checkpoints", str(out_file.absolute()), "--run-output-path", str(data_output_dir)])
with open(data_output_dir / (out_file.stem + "_metrics.json")) as metrics_file:
    metrics_dict = json.load(metrics_file)

metrics = {}
for metric_name, metric_val in metrics_dict.items():
    if type(metric_val) in [float, int]:
        metrics[f"test/{metric_name}"] = metric_val

logger = TrainingLogger(tensorboard_dir)

logger.add_hparams(hp_dict, metrics)
