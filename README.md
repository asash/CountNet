This is the source code implementing the paper **CountNet: Utilising Repetition Counts in Sequential Recommendation** published in the proceedings of 47$^th$ European Conference on Information Retrieval. 

## Authors:
Aleksandr V. Petrov, University of Glasgow (work done while in Amazon.com)
Efi Karra, Amazon.com
Sean Murphy, Amazon.com

```
@inproceedings{petrov2025countnet,
  title={CountNet: Utilising Repetition Counts in Sequential Recommendation},
  author={Petrov, Aleksandr Vladimirovich and Karra, Efi and Murphy, Sean},
  booktitle={Proceedings of the 17th ACM Conference on Recommender Systems},
  pages={116--128},
  year={2023}
}
```


## Structure

The source code is in the `src` directory.

1. The implementation of the sequential recommender and training methods is in `src/recommenders`.
    - `src/recommenders/SASRec.py` contains the SASRec recommender, and `src/recommenders/utils/sequencer.py` contains
      the sequencer used for padding/truncation of each user's item sequence.
    - The logits modifications of CountNet described in this paper are in `src/recommenders/logit_transform/logit_transform.py`.

2. `src/training/train_sasrec.py`,`src/training/train_sasrec_countnet.py` and `src/sagemaker/sagemaker_training.py` are used
   to launch jobs locally and on AWS Sagemaker.

## Datasets and Processing

We use Booking, Gowalla and Taobao, popular open datasets for sequential recommendations.

1. Download Taobao from `https://tianchi.aliyun.com/dataset/53` and place it in `data/raw/taobao`. 
2. Download the `train_set.csv` and `test_set.csv` from `https://github.com/bookingcom/ml-dataset-mdt` and place it
   in `data/raw/booking`. 
3. Download Gowalla dataset from `https://snap.stanford.edu/data/loc-gowalla.html` and place it in `data/raw/gowalla`.
4. Run the scripts in `src/preprocess` to pre-process the datasets. The preprocessed data will be stored `data/processed/`.

## Usage

Note that the PYTHONPATH for your conda environment should be set to the base of this repository.

### Local Run

Use `python src/training/train_sasrec.py` to schedule local runs, this script has default values for the training
parameters.

### Launching Jobs on Sagemaker

You can launch jobs on Sagemaker via `python src/sagemaker/sagemaker_training.py <config path>` with the appropriate
config path. The configs for all the jobs in our experiments are available in `src/sagemaker/training_configs`, e.g.

`python src/sagemaker/sagemaker_training.py 'src/sagemaker/training_configs/booking/sasrec-backbone/sasrec-countnet.json'`

--- 
