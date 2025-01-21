import torch.utils
from src.recommenders.logit_transform.rerank_net import RerankNet
from src.recommenders.logit_transform.logit_transform import LogitTransform
from src.recommenders.utils.sequencer import Sequencer
from src.recommenders.sequential_model import SequentialModel
from src.recommenders.sequential_recommender import SequentialRecommender, SequentialRecommenderConfig
import torch


class SASRecCountNetConfig(SequentialRecommenderConfig):
    def __init__(self, embedding_size=256, num_layers=3, nhead=4, dim_feedforward=1024, dropout=0.5,
                 count_transform="id",
                 logit_aggregate ="replace",
                 rerank_cutoffs="",
                 logit_item_repr ="embeddings",
                 backbone = "SASRec",
                 separate_logit_projection = False,
                 beta = 1.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout=dropout
        self.count_transform = count_transform
        self.logit_aggregate = logit_aggregate
        self.logit_item_repr = logit_item_repr
        self.separate_logit_projection = separate_logit_projection
        self.rerank_cutoffs = [int(num) for num in rerank_cutoffs.split(",")] if rerank_cutoffs != "" else []
        self.backbone = backbone
        self.beta = beta


class SASRecCountNetModel(SequentialModel):
    def __init__(self, num_items, config: SASRecCountNetConfig):
        super().__init__()
        self.position_embeddings = torch.nn.Embedding(config.sequence_len, config.embedding_size)
        self.item_embeddings = torch.nn.Embedding(num_items, config.embedding_size)
        self.unembeddings = torch.nn.Embedding(num_items, config.embedding_size)
        self.backbone = config.backbone
        self.emb_dropout = torch.nn.Dropout(config.dropout)
        self.rerank_cutoffs = config.rerank_cutoffs

        if self.backbone == "SASRec":
            transformer_encoder_layer = torch.nn.TransformerEncoderLayer(config.embedding_size, config.nhead, config.dim_feedforward, config.dropout, batch_first=True, norm_first=True)
            self.transformer = torch.nn.TransformerEncoder(transformer_encoder_layer, num_layers=config.num_layers)
            self.num_items = num_items
            self.output_norm = torch.nn.LayerNorm(config.embedding_size)

        elif self.backbone == "GRU":
            self.gru = torch.nn.GRU(input_size=config.embedding_size, hidden_size=config.embedding_size, num_layers=config.num_layers, dropout=config.dropout, batch_first=True)


        self.separate_logit_projection = config.separate_logit_projection
        self.logittransform = LogitTransform(num_items, input_output_dim=config.embedding_size,
                                             counts_transform_func=config.count_transform,
                                             logit_item_repr=config.logit_item_repr,
                                             sequence_len = config.sequence_len,
                                             beta=config.beta
                                             )
        self.logit_aggregte = config.logit_aggregate
        if self.separate_logit_projection:
            self.logit_projection = self.create_projection(input_output_dim=config.embedding_size)

        if len(config.rerank_cutoffs) > 0:
            self.rerank_net = RerankNet(num_items, config.embedding_size, config.rerank_cutoffs)

    def create_projection(self, input_output_dim):
        return torch.nn.Linear(input_output_dim, input_output_dim, bias=True)

    def split_training_batch(self, batch):
        item_ids = batch["sequences"]
        inputs = item_ids[:,:-1]  # Batch x Sequence -1 x Embedding
        labels = item_ids[:,1:].unsqueeze(-1)
        padding_mask = batch["padding_mask"][:,:-1]
        pass

        position_ids = batch["position_ids"][:-1] + 1 # shift one left
        return {"sequences": inputs, "labels": labels, "padding_mask": padding_mask, "position_ids": position_ids}

    def forward(self, batch):
        train_batch = self.split_training_batch(batch)
        hidden, embedding_weights = self.hidden_state(train_batch)

        logits_transform = self.logittransform(train_batch["sequences"], hidden, embedding_weights)
        if self.separate_logit_projection:
            hidden = self.logit_projection(hidden)

        logits = torch.einsum("bse, ie -> bsi", hidden, embedding_weights)
        logits = self.aggregate(logits, logits_transform, hidden, embedding_weights)

        logprobs = torch.log_softmax(logits, -1).gather(2, train_batch["labels"]).squeeze(-1)

        non_masked = 1-train_batch["padding_mask"]

        loss = -(logprobs * non_masked).sum(axis=-1)/non_masked.sum(axis=-1)
        mean_loss = loss.mean()

        return {
            "loss": mean_loss
        }

    def hidden_state(self, batch):
        item_embeddings = self.item_embeddings(batch["sequences"])
        if self.backbone == "SASRec":
            position_embeddings = self.position_embeddings(batch["position_ids"]).unsqueeze(0)
            transformer_input = item_embeddings + position_embeddings
            transformer_input = self.emb_dropout(transformer_input)

            causality_mask = torch.nn.Transformer.generate_square_subsequent_mask(batch["sequences"].shape[1], device=batch["sequences"].device)
            transformer_output = self.transformer(is_causal=True, src_key_padding_mask = batch["padding_mask"], mask=causality_mask,src=transformer_input)
            transformer_output = self.output_norm(transformer_output)

        elif self.backbone == "GRU":
            item_embeddings = self.emb_dropout(item_embeddings)
            transformer_output = self.gru(item_embeddings)[0]

        elif self.backbone == "FMC":
            transformer_output = self.emb_dropout(item_embeddings)

        embedding_weights = self.emb_dropout(self.unembeddings.weight)
        return transformer_output,embedding_weights

    def aggregate(self, logits, logits_transform, hidden, item_emb):
        if len(self.rerank_cutoffs) > 0:
            logits = self.rerank_net.forward(hidden, logits, item_embeddings=item_emb)

        if self.logit_aggregte == "replace":
            logits_transform_mask = (logits_transform != 0).to(torch.float)
            result = logits_transform_mask * logits_transform + (1 - logits_transform_mask) * logits
            return result
        else:
          raise ValueError(f"logit_aggregte {self.logit_aggregte} not supported")

    def predict(self, batch):
        hidden, embedding_weights = self.hidden_state(batch)
        logits_transform  = self.logittransform(batch["sequences"], hidden, embedding_weights)
        logits_transform = logits_transform[:, -1 , :]

        hidden = hidden[:,-1, :]
        if self.separate_logit_projection:
            hidden = self.logit_projection(hidden)

        logits = torch.einsum("be, ie -> bi", hidden, embedding_weights)
        return self.aggregate(logits, logits_transform, hidden, embedding_weights)



class SASRecCountNetRecommender(SequentialRecommender):
    def __init__(self, config: SASRecCountNetConfig) -> None:
        super().__init__(config)
        self.config = config

    def init_model(self, sequencer: Sequencer) -> SequentialModel:
        return SASRecCountNetModel(num_items=len(sequencer.item_mapping), config=self.config).to(self.config.device)
