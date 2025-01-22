# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


class RerankNet(torch.nn.Module):
    def __init__(self, num_items, d_model, rerank_cutoffs):
        super().__init__()
        self.num_items = num_items
        self.rerank_cutoffs = sorted(rerank_cutoffs)
        self.d_model = d_model
        modules = []
        for i in range(len(rerank_cutoffs)):
            modules.append(torch.nn.Linear(d_model, d_model, bias=True))
        self.projections = torch.nn.ModuleList(modules)

    def get_items_to_rerank(self, topk_indices, prev_cutoff, cutoff):
        if len(topk_indices.shape) == 3:
            return topk_indices[:, :, prev_cutoff:cutoff]
        elif len(topk_indices.shape) == 2:
            return topk_indices[:, prev_cutoff:cutoff]

    def compute_logit(self, rerank_hidden, rerank_item_embs):
        if len(rerank_hidden.shape) == 3:
            return torch.einsum("bse, bsne -> bsn", rerank_hidden, rerank_item_embs)
        else:
            return torch.einsum("be, bne -> bn", rerank_hidden, rerank_item_embs)

    def forward(self, hidden_states, logits, item_embeddings):
        topk_logits = torch.topk(logits, self.rerank_cutoffs[-1])
        topk_indices = topk_logits.indices
        prev_cutoff = 0
        all_new_logits = []
        for i in range(len(self.rerank_cutoffs)):
            cutoff = self.rerank_cutoffs[i]
            items_to_rerank = self.get_items_to_rerank(topk_indices, prev_cutoff, cutoff)
            reranker = self.projections[i]
            rerank_hidden = reranker(hidden_states)
            rerank_item_embs = item_embeddings[items_to_rerank]
            updated_item_logits = self.compute_logit(rerank_hidden, rerank_item_embs)
            all_new_logits.append(updated_item_logits)
            prev_cutoff = cutoff
        all_new_logits = torch.cat(all_new_logits, -1)
        return logits.scatter_(len(topk_indices.shape) - 1, topk_indices, all_new_logits)
