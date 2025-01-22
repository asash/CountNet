# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
import json
from pathlib import Path
import dill
import ir_measures
import pandas as pd
from src.utils.load_data import load_data
from src.utils.project_dirs import get_dataset_checkpoints_dir, get_recs_dir
from src.utils.ir_measures_converters import get_irmeasures_qrels, get_irmeasures_run
from src.recommenders.recommender import Recommender
from ir_measures import nDCG, R


parser = ArgumentParser()
parser.add_argument("--checkpoints", nargs="+", required=True)
parser.add_argument("--names", nargs="+", required=False, default=None)
parser.add_argument("--top-k", type=int, default=10)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--absolute-path", type=bool, default=False)
parser.add_argument("--run-output-path", type=str, default=None)
args = parser.parse_args()


dataset_checkpoints_dir = get_dataset_checkpoints_dir(args.dataset)

if not args.absolute_path:
    checkpoint_paths=[dataset_checkpoints_dir/chkp for chkp in args.checkpoints]
else:
    checkpoint_paths=[Path(chkp) for chkp in args.checkpoints]

dataset = args.dataset
data = load_data(dataset)
if args.run_output_path is None:
    recs_folder = get_recs_dir(dataset)
else:
    recs_folder = Path(args.run_output_path)
    recs_folder.mkdir(exist_ok=True, parents=True)


def get_recs(checkpoint_path):
  recommender:Recommender = dill.load(open(checkpoint_path, "rb"))
  user_ids = list(data['test'].user_id)
  recs = recommender.recommend(user_ids=user_ids, top_k=args.top_k)
  return recs

qrels = get_irmeasures_qrels(data['test'])
metrics = [nDCG@10, R@10, R@1]

evaluator = ir_measures.evaluator(metrics, qrels)

if args.names is not None:
    assert(len(args.names) == len(args.checkpoints))
    names = args.names
else:
    names = [checkpoint.stem for checkpoint in checkpoint_paths]


results = []
for checkpoint_path, name in zip(checkpoint_paths, names):
    checkpoint_name = checkpoint_path.stem
    irmeasures_run_file = recs_folder / (f"{checkpoint_name}_top_{args.top_k}_irm_run.csv")
    if not (checkpoint_path.exists() or irmeasures_run_file.exists()) :
        raise AttributeError(f"unknown checkpoint {checkpoint_paths}")
    if not irmeasures_run_file.exists():
        recs = get_recs(checkpoint_path)
        run: pd.DataFrame = get_irmeasures_run(recs, data["test"])
        run.to_csv(irmeasures_run_file, index=False)
    else:
        run = pd.read_csv(irmeasures_run_file)
        run["query_id"] = run.query_id.astype('str')
        run["doc_id"] = run['doc_id'].astype('str')
    result = evaluator.calc_aggregate(run)

    result["Model"] = name

    jsonable_result = {str(k): v for (k, v) in result.items()}
    metrics_file = recs_folder / (f"{checkpoint_name}_metrics.json")
    with open(metrics_file, "w") as out:
        json.dump(jsonable_result, out, indent=4)
    results.append(result)

print(f"evaluation results for {dataset}")
df = pd.DataFrame(results)
df = df.set_index("Model")
print(df)






