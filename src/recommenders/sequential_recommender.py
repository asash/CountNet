import pandas as pd
import time
from src.recommenders.sequential_model import SequentialModel
from src.recommenders.recommender import Recommender
from src.recommenders.utils.sequencer  import Sequencer
from ir_measures import nDCG
from torch.utils.data import DataLoader
from src.recommenders.utils.logger import TrainingLogger
from copy import deepcopy
import torch
import tqdm
import ir_measures
from collections import defaultdict, deque

from src.utils.ir_measures_converters import get_irmeasures_qrels, get_irmeasures_run


class SequentialRecommenderConfig(object):
    def __init__(self, sequence_len=200, batch_size=512, device='cpu:0',
                 max_epoch=100000, batches_per_epoch=128,
                 val_metric=nDCG, val_topk=10,
                early_stop_patience_epochs=200,
                effective_batch_size = 512
                ):

        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.device = device
        self.max_epochs = max_epoch
        self.batches_per_epoch = batches_per_epoch
        self.val_metric = val_metric
        self.val_topk = val_topk
        self.early_stop_patience_epochs = early_stop_patience_epochs
        self.effective_batch_size = effective_batch_size


class SequentialRecommender(Recommender):
    def __init__(self, config: SequentialRecommenderConfig) -> None:
        self.sequencer:Sequencer = None
        self.model: SequentialModel = None #lazy init with init_model
        self.config: SequentialRecommenderConfig = config
        if config.effective_batch_size % config.batch_size != 0:
            raise AttributeError("Effective batch size should be a multiple of batch size")

    def init_model(self, sequencer: Sequencer) -> SequentialModel:
        raise NotImplementedError


    def train(self, train_actions, val_actions, all_val_actions, tensorboard_dir) -> None:
        torch.autograd.set_detect_anomaly(True)
        self.sequencer = Sequencer(train_actions, max_length=self.config.sequence_len)
        self.model = self.init_model(self.sequencer)
        logger = TrainingLogger(tensorboard_dir)

        all_user_ids = train_actions.user_id.unique()
        train_loader = DataLoader(all_user_ids, collate_fn=self.sequencer.get_sequences_batch, batch_size=self.config.batch_size, shuffle=True)

        best_weights = deepcopy(self.model.state_dict())
        best_epoch = 0
        best_loss_epoch = 0

        best_val_result = float("-inf")
        best_loss = float("+inf")
        batches_per_update = self.config.effective_batch_size // self.config.batch_size
        batches_per_epoch = self.config.batches_per_epoch * batches_per_update
        print(f"Gradient accumulation for {batches_per_update} batches")

        loss_hist = deque(maxlen=batches_per_epoch)
        train_losses = defaultdict(lambda: deque(maxlen=batches_per_epoch))

        optimiser = torch.optim.Adam(self.model.parameters())
        qrels = get_irmeasures_qrels(val_actions)
        evaluator = ir_measures.evaluator([self.config.val_metric], qrels)
        start_time = time.time()

        for epoch in range(1, self.config.max_epochs+1):
            print(f"==== epoch: {epoch} ====")
            self.model.train()
            pbar = tqdm.tqdm(total=batches_per_epoch, ncols=70, ascii=True)
            batches_processed = 0
            while batches_processed < batches_per_epoch:
                for batch in train_loader:
                    if batches_processed >= batches_per_epoch:
                        break
                    batch_on_device = {k:v.to(self.config.device) for (k,v) in batch.items()}
                    result = self.model.forward(batch_on_device)
                    ce_all = result['loss']
                    loss = ce_all
                    loss.backward()
                    batches_processed += 1
                    if batches_processed % batches_per_update == 0:
                        optimiser.step()
                        optimiser.zero_grad()
                    loss_hist.append(loss.item())
                    for loss in result:
                        train_losses[loss].append(result[loss].detach().item())
                    pbar.update(1)
                    pbar.set_description(f"mean loss: {sum(loss_hist)/len(loss_hist):0.6g}")
            pbar.close()

            val_result = self.validate(val_actions, evaluator)
            val_metric_value = val_result[self.config.val_metric]

            val_loss = val_result["val_loss"]
            if val_loss < best_loss:
                best_loss_epoch = epoch
                best_loss = val_loss

            if val_metric_value > best_val_result:
                 print(f"New best val ndcg. Updating best weights...")
                 best_weights = deepcopy(self.model.state_dict())
                 best_val_result = val_metric_value
                 best_epoch = epoch


            metric_name = str(self.config.val_metric)
            print(f"Val {str(self.config.val_metric)}: {val_metric_value:0.4g} (best: {best_val_result:0.4g})")
            print(f"Val loss: {val_loss:0.6g} (best: {best_loss:0.6g})")
            print(f"Train loss: {sum(loss_hist)/len(loss_hist):0.6g}")

            logger.add_scalar("loss/val", val_loss, epoch)
            logger.add_scalar("loss/best_val", best_loss, epoch)
            logger.add_scalar(f"val_{metric_name}/all", val_metric_value, epoch)
            logger.add_scalar(f"val_crossentropy/all", val_result["crossentropy_all"], epoch)
            logger.add_scalar(f"val_{metric_name}/best_val", best_val_result, epoch)
            logger.add_scalar(f"loss/train", sum(loss_hist)/len(loss_hist), epoch)

            for loss_component in train_losses:
                logger.add_scalar("train_loss_components/" + loss_component, sum(train_losses[loss_component])/len(train_losses[loss_component]), epoch)

            metric_no_improvement = epoch - best_epoch
            metric_patience = max(self.config.early_stop_patience_epochs - metric_no_improvement, 0)

            loss_no_improvement = epoch - best_loss_epoch
            loss_patience = max(self.config.early_stop_patience_epochs - loss_no_improvement, 0)
            patience = max(metric_patience, loss_patience)

            print(f"Patience: {patience} ({metric_patience} {metric_name}; {loss_patience} loss)")
            logger.add_scalar(f"patience/{metric_name}", metric_patience, epoch)
            logger.add_scalar("patience/loss", loss_patience, epoch)
            logger.add_scalar("patience/overall", patience, epoch)

            training_time = int(time.time() - start_time)
            hours, remainder = divmod(training_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print('Training time: {:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)))
            logger.tb_writer.flush()
            if patience <= 0:
                print("No patience left. Early stopping after epoch {epoch}")
                break
        print(f"loading best weights from epoch {best_epoch}")
        self.model.load_state_dict(best_weights)
        self.sequencer.add_val_actions(all_val_actions)
        del(train_loader)

    def validate(self, val_actions, evaluator):
        result = []
        self.model.eval()
        val_userids = list(val_actions.user_id)
        val_itemids = list(val_actions.item_id)
        start = 0
        gt_logprobs = []
        gt_is_predictable = []
        while start < len(val_itemids):
            batch_userids = val_userids[start:start+self.config.batch_size]
            batch_itemids = val_itemids[start:start+self.config.batch_size]
            batch = self.sequencer.get_sequences_batch(batch_userids, shortening=1)
            gt_items = torch.tensor([self.sequencer.item_mapping.get(item_id, self.sequencer.unknown_item_id) for item_id in batch_itemids], device=self.config.device, dtype=torch.int64).unsqueeze(-1)
            unknown_item_mask  =  (gt_items != int(self.sequencer.unknown_item_id)).to(torch.float32)
            batch_on_device = {k:v.to(self.config.device) for (k,v) in batch.items()}
            logits = self.model.predict(batch_on_device)
            logits[:, :self.sequencer.num_special_items] = -1e9
            logprobs = logits.log_softmax(-1).gather(1, gt_items)

            top_k_recs = torch.topk(logits, self.config.val_topk)
            gt_logprobs.append(logprobs.detach().squeeze(-1))
            gt_is_predictable.append(unknown_item_mask.detach().squeeze(-1))
            decoded_recs = self.sequencer.decode_topk(top_k_recs)
            start += self.config.batch_size
            result += decoded_recs
        gt_logprobs = torch.cat(gt_logprobs)
        is_predictable_mask = torch.cat(gt_is_predictable)
        eps = 1e-9

        ce_all = -torch.sum(gt_logprobs * is_predictable_mask) / (torch.sum(is_predictable_mask) + eps)

        val_run: pd.DataFrame = get_irmeasures_run(result,val_actions)
        eval_result = evaluator.calc_aggregate(val_run)

        eval_result['val_loss']  = ce_all.detach().item()
        eval_result['crossentropy_all'] = ce_all.detach().item()
        return eval_result


    def recommend(self, user_ids, top_k):
        result = []
        self.model.eval()
        start = 0
        while start < len(user_ids):
            batch_userids = user_ids[start:start+self.config.batch_size]
            batch = self.sequencer.get_sequences_batch(batch_userids, shortening=1)
            batch_on_device = {k:v.to(self.config.device) for (k,v) in batch.items()}
            logits = self.model.predict(batch_on_device)
            logits[:, :self.sequencer.num_special_items] = -float("inf")
            top_k_recs = torch.topk(logits, top_k)
            decoded_recs = self.sequencer.decode_topk(top_k_recs)
            result += decoded_recs
            start += self.config.batch_size
        return result
