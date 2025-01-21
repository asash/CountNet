import torch
from src.recommenders.utils.sequencer import Sequencer


class LogitTransform(torch.nn.Module):
    def __init__(self, num_items, sequence_len,
                 input_output_dim = 128,
                 counts_transform_func = "logPlusOne",
                 logit_item_repr ="embeddings",  # <embeddings - use model's item embeddings; hidden - use average of hidden representations; both - use both>
                 beta = 1,  #only used in logPlusBetaHyper
                 ):
        super().__init__()
        self.item_repr = logit_item_repr
        self.beta = beta
        self.sequence_len = sequence_len

        if self.use_emb_net() :
            self.embeddings_net = self.create_projection(input_output_dim)

        if self.use_pointer_net:
            self.pointer_net = self.create_projection(input_output_dim)

        self.counts_transform_func = counts_transform_func
        self.num_items = num_items

        if self.counts_transform_func == "firstAlphaSecondBeta":
            self.alpha = torch.nn.Parameter(torch.tensor(1.0))
            self.beta_raw = torch.nn.Parameter(torch.tensor(0.5413))

    def use_emb_net(self):
        return self.item_repr in ["embeddings", "both"]

    def use_pointer_net(self):
        return self.item_repr in ["hidden", "both"]

    def create_projection(self, input_output_dim):
        return torch.nn.Linear(input_output_dim, input_output_dim, bias=True)

    def transform_counts(self, counts, input_seq):
        if self.counts_transform_func == "id":
            return counts

        if self.counts_transform_func == "firstAlphaSecondBeta":
            first = (counts == 1) * self.alpha
            mask_cnt_2 = counts < 2
            beta = torch.nn.functional.softplus(self.beta_raw)
            result = torch.log(torch.nn.functional.relu(counts - 1) + beta) # beta  in the paper equals beta +1 here
            result = torch.where(mask_cnt_2, torch.tensor(0.0, dtype = result.dtype), result)
            result = result + first
            mask = counts == 0
            mask_1 = (input_seq <= Sequencer.START_ITEM_ID).unsqueeze(1)
            mask_2 = (input_seq <= Sequencer.START_ITEM_ID).unsqueeze(2)

            result = torch.where(mask, torch.tensor(0.0, dtype=result.dtype), result)
            result = torch.where(mask_1, torch.tensor(0.0, dtype=result.dtype), result)
            result = torch.where(mask_2, torch.tensor(0.0, dtype=result.dtype), result)

            return result

        if self.counts_transform_func == "logPlusOne":
            with  torch.no_grad():
                counts = torch.log2(counts + 1)
                return counts

        if self.counts_transform_func == "logPlusBetaHyper":
            with  torch.no_grad():
                counts = torch.log2(counts + self.beta)
                return counts

        if self.counts_transform_func == "zero":
            with  torch.no_grad():
                counts = torch.zeros_like(counts)
                return counts

    def forward(self, input_seq, hidden_states, item_embeddings):
        reps_expanded_1 = input_seq.unsqueeze(2)
        reps_expanded_2 = input_seq.unsqueeze(1)
        repetitions = (reps_expanded_1 == reps_expanded_2).to(torch.int32)
        counts = repetitions.cumsum(dim=1)

        selected_embs = item_embeddings[input_seq]
        logits_nocount = self.get_logits_no_counts(hidden_states, selected_embs, repetitions, counts)

        logits_transformed = logits_nocount * self.transform_counts(counts, input_seq)
        result = torch.zeros(size=(input_seq.shape[0], input_seq.shape[1], self.num_items), dtype=logits_transformed.dtype, device=logits_transformed.device)
        index = input_seq.unsqueeze(1).expand_as(logits_transformed)
        result = result.scatter_reduce(2, index, logits_transformed, "mean", include_self=False)
        return result

    def get_logits_no_counts(self, hidden_states, selected_embs, repetitions, counts):
        logits_from_emb = 0
        logits_from_hidden = 0
        if self.use_emb_net():
            emb = self.embeddings_net(hidden_states)
            logits_from_emb = torch.einsum("bse, bne -> bsn", emb, selected_embs)

        sequence_length = hidden_states.shape[1]
        if self.use_pointer_net():

            with torch.no_grad():
                triangular_mask = torch.tril(torch.ones((sequence_length, sequence_length), dtype=torch.int32, device=hidden_states.device), diagonal=0)
                masked_repetitions = repetitions * triangular_mask
                idx = torch.arange(1, repetitions.shape[1] +1, device=masked_repetitions.device).unsqueeze(0).unsqueeze(-1)
                seen_at_idx = (repetitions * idx)
                last_seen = torch.cummax(seen_at_idx, dim=1).values - 1
                last_seen = last_seen.unsqueeze(-1).expand(-1, -1, -1, hidden_states.shape[-1])
                last_seen_mask = last_seen < 0
                last_seen = torch.relu(last_seen).to(torch.int64)

            hidden_rep_sum_upto_pos =  torch.einsum("bks, bse-> bke", masked_repetitions.float(), hidden_states).unsqueeze(-2).expand(-1, -1, sequence_length, -1)
            sum_from_last = torch.gather(hidden_rep_sum_upto_pos, dim=1, index=last_seen)
            counts_expanded = counts.unsqueeze(-1)
            counts_zero_mask = counts_expanded == 0
            counts_expanded = torch.where(counts_zero_mask, torch.tensor(1.0, dtype=counts_expanded.dtype), counts_expanded)
            avg = sum_from_last.div(counts_expanded)  * (1-counts_zero_mask.to(torch.float32))

            embs_from_hidden = torch.where(last_seen_mask, torch.tensor(0.0, dtype=avg.dtype), avg)
            pointer_hidden = self.pointer_net(hidden_states)
            logits_from_hidden = torch.einsum("bse, bsne -> bsn", pointer_hidden, embs_from_hidden)
        return logits_from_emb + logits_from_hidden


