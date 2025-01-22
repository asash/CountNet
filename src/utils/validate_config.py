# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import ir_measures


def get_config_dict(config):
    config_dict = copy.deepcopy(config.__dict__)
    for key, val in config_dict.items():
        if issubclass(type(val), ir_measures.measures.Measure):
            val = str(val)
            config_dict[key] = val
        if key == "rerank_cutoffs" and type(val) == list:
            val = ",".join([str(cutoff) for cutoff in config_dict[key]])
            config_dict[key] = val
        if not(type(val) in (str, int, float, bool)):
            raise AttributeError(f"Wrong type for {key}: {type(val)}")
    return config_dict
