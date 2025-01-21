#converts train data to sequential format
#Also, handles the item id representations
#Uses item <0> as padding

from collections import defaultdict
import torch


class Sequencer(object):
    PAD_ITEM_ID = 0
    START_ITEM_ID = 1
    UNKNOWN_ITEM_ID= 2
    MASK_ITEM_ID = 3

    def __init__(self, training_data, max_length=200):
        self.pad_item_id = self.PAD_ITEM_ID
        self.start_item_id = self.START_ITEM_ID
        self.unknown_item_id = self.UNKNOWN_ITEM_ID
        self.mask_item_id = self.MASK_ITEM_ID

        self.item_mapping_reverse = {self.pad_item_id: "<PAD>"}

        self.item_mapping = {"<PAD>": self.pad_item_id,
                             "<START>": self.start_item_id,
                             "<UNKNOWN>": self.unknown_item_id,
                             "<MASK>": self.mask_item_id
                             }
        self.num_special_items = len(self.item_mapping)

        all_items = sorted(training_data.item_id.unique())
        for num, id in enumerate(all_items):
            external_id = id
            internal_id = num + self.num_special_items
            self.item_mapping_reverse[internal_id] = external_id
            self.item_mapping[external_id] = internal_id
        user_sequences = defaultdict(list)
        last_ts = -1000
        self.max_length=max_length
        for _, user_id, item_id, timestamp in training_data[["user_id", "item_id", "timestamp"]].itertuples():
            if timestamp < last_ts:
                raise ValueError("train data not sorted by timestamp")
            user_sequences[user_id].append(self.item_mapping[item_id])
        self.user_sequences = dict(user_sequences)

    #  call this after model training finished, so that val actions are used for inference
    def add_val_actions(self, all_val_actions):
        for _, user_id, item_id, timestamp in all_val_actions[["user_id", "item_id", "timestamp"]].itertuples():
            if item_id in self.item_mapping:
                self.user_sequences[user_id].append(self.item_mapping[item_id])
        pass


    def get_sequences_batch(self, user_ids, shortening=0):
        sequences = [self.user_sequences[user_id] for user_id in user_ids]
        longest = 0
        for i in range(len(user_ids)):
            end = len(sequences[i])
            start = max(0, end-self.max_length + 1 + shortening)
            sequences[i] = [self.start_item_id] + sequences[i][start:end]
            longest = max(len(sequences[i]), longest)
        #  left pad. we want the newst items to be the latest in the sequence
        for i in range(len(sequences)):
            sequences[i] = [0] * (longest - len(sequences[i])) + sequences[i]
        sequences = torch.tensor(sequences)
        padding_mask = (sequences == self.pad_item_id).to(torch.float32)
        last_position_id = self.max_length
        first_position_id = last_position_id - longest
        position_ids = torch.arange(first_position_id, last_position_id)
        return {"sequences": sequences,
                "position_ids": position_ids,
                "padding_mask": padding_mask}

    def decode_topk(self, encoded_topk: torch.return_types.topk):
        internal_item_ids: torch.Tensor = encoded_topk.indices
        internal_item_ids = internal_item_ids.detach().cpu().numpy()
        res = []
        for row in range(len(internal_item_ids)):
            res_row = []
            for col in range(len(internal_item_ids[row])):
                internal_id = internal_item_ids[row][col]
                score = encoded_topk.values[row][col].item()
                external_id = self.item_mapping_reverse[internal_id]
                res_row.append((external_id, score))
            res.append(res_row)
        return res
