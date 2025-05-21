import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import pandas as pd
import lightning as L
from typing import Literal
from torch import Tensor
from copy import deepcopy
from collections import defaultdict
from torch.utils.data import default_collate
from torchmetrics.functional.classification import multiclass_accuracy, binary_accuracy
from torchmetrics.functional.text import bleu_score, rouge_score, word_error_rate
from lightning.pytorch.utilities import rank_zero_only
from transformers import AutoTokenizer, T5ForConditionalGeneration, BartForConditionalGeneration, get_cosine_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput

from .modules import PromptEmbedder, EEGEncoder, Aligner



class GLIM(L.LightningModule):

    SUPPORTED_TEXT_MODELS = Literal["google/flan-t5-xl", "google/flan-t5-large", 
                                    "facebook/bart-large-cnn", "jbochi/madlad400-3b-mt",]

    def __init__(self, 
                 input_eeg_len = 1280,
                 hidden_eeg_len = 96,
                 input_text_len = 96,
                 tgt_text_len = 64,
                 input_dim = 128,
                 hidden_dim = 128,
                 embed_dim = 1024,
                 text_model_id: SUPPORTED_TEXT_MODELS = "google/flan-t5-large", 
                 prompt_nums: tuple[int] = (3, 3, 31),
                 prompt_dropout_probs: tuple[float] = (0.0, 0.0, 0.0),
                 evaluate_prompt_embed: Literal['zero', 'sum', 'mean', 'src'] = 'src',
                 n_in_blocks: int = 6,
                 n_out_blocks: int = 6,
                 in_temporal_modulate: bool = True,
                 out_is_causal: bool = True,
                 prompt_tuning_len: bool = 0,
                 num_heads = 8,
                 mlp_ratio = 4,
                 dropout = 0.0,
                 clip_loss_weight = 0.5,
                 commitment_loss_weight = 0.0,
                 commitment_loss_key: Literal['mse','kl_div']= 'mse',
                 use_y_mask = False,
                 bsz_train = 48,
                 bsz_val = 24,
                 lr = 1e-5,
                 weight_decay = 0,
                 full_val_interval = 10,
                 bs_retrieval = 24,
                ):
        
        super().__init__()

        self.input_text_len = input_text_len
        self.tgt_text_len = tgt_text_len
        self.prompt_tuning_len = prompt_tuning_len
        self.eval_pembed = evaluate_prompt_embed

        self.λ = clip_loss_weight
        self.ε = commitment_loss_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.bsz_train = bsz_train
        self.bsz_val = bsz_val
        self.full_val_interval = full_val_interval
        self.bsz_retrieval = bs_retrieval
        self.prompt_keys = {
            # 'task': ['<Normal Reading>'] + ['<Relation Extraction>', '<Sentiment Classification>',],
            'task': ['<UNK>'] + ['<NR>', '<TSR>'],
            # 'task': ['<NR>'] + ['<SC>', '<RE>'],
            'dataset': ['<UNK>'] + ['ZuCo1', 'ZuCo2',],
            'subject': ['<UNK>'] + ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 
                                    'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH', 
                                    'YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 
                                    'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS', 
                                    'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL',],
            }
        # for logging and classification
        self.raw_task_keys = ['task1', 'task2', 'task3']
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        self.relation_labels = ['awarding', 'education', 'employment',
                                'foundation', 'job title', 'nationality', 
                                'political affiliation','visit', 'marriage']
        
        self.p_embedder = PromptEmbedder(input_dim, 
                                         prompt_nums, prompt_dropout_probs, self.prompt_keys)
        # self.task_embed_proj = nn.Linear(input_dim, embed_dim * prompt_tuning_len)

        self.eeg_encoder = EEGEncoder(input_eeg_len, hidden_eeg_len, input_dim, hidden_dim, 
                                      prompt_tuning_len, n_in_blocks, n_out_blocks, 
                                      in_temporal_modulate, out_is_causal, 
                                      num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        
        self.aligner = Aligner(hidden_dim, embed_dim, num_heads, dropout, commitment_loss_key, use_y_mask) 
        self.use_y_mask = use_y_mask
        self.text_model_id = text_model_id
        self.embed_dim = embed_dim

        self.save_hyperparameters(logger=True)

    def setup(self, stage):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_id)
        self.text_model = T5ForConditionalGeneration.from_pretrained(
            self.text_model_id, device_map = self.device,
            torch_dtype = torch.bfloat16, # FIXME
            ).requires_grad_(False)
        assert self.embed_dim == self.text_model.config.d_model

    def add_prompt(self, on:Literal['task','dataset','subject'], prompt):
        # TODO: on x -> self.x_prompts += prompt
        # self.p_embedder.
        # maybe self.prompt_nums
        pass

    def on_save_checkpoint(self, checkpoint: torch.Dict[str, torch.Any]) -> None:
        for key in deepcopy(list(checkpoint['state_dict'].keys())):
            if 'text_model' in key: 
                checkpoint['state_dict'].pop(key)
        
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad == True]
        opt = torch.optim.Adam(params, 
                               lr = self.lr,
                               weight_decay = self.weight_decay)
        # lr_scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, 
        #                                                num_training_steps=self.trainer.max_epochs)
        # return {"optimizer": opt,
        #         "lr_scheduler": lr_scheduler}
        return opt
    
    def tokenize(self, texts: list[str], max_length: int) -> tuple[torch.Tensor]:
        inputs = self.tokenizer(texts, max_length=max_length, padding='max_length', 
                                truncation=True, return_tensors="pt") 
        ids = inputs['input_ids'].to(self.device)
        mask = inputs['attention_mask'].to(self.device)
        return ids, mask
    
    def encode_labels(self, labels:list[str], ignore_idx=-1):
        label_ids = []
        for label in labels:
            if label in self.relation_labels:
                label_id = self.relation_labels.index(label)
            elif label in self.sentiment_labels:
                label_id = self.sentiment_labels.index(label)
            else:
                assert label == 'nan'
                label_id = ignore_idx
            label_ids.append(label_id)
        label_ids = torch.tensor(label_ids, dtype=torch.int, device=self.device)
        return label_ids

    def get_inputs(self, batch):
        eeg = batch['eeg']        # (n, l, c)
        eeg_mask = batch['mask']  # (n, l)    1 for unmasked; 0 for masked
        prompts = batch['prompt']  # NOTE: [tuple('task'), tuple('dataset'), tuple('subject')] after collate
        input_text = batch['input text']        # list[str]
        tgt_text = batch['target text']         # list[str]
        # for logging and cal metrics
        sentiment_label = batch['sentiment label']      # list[str]
        relation_label = batch['relation label']      # list[str]
        raw_task_key = batch['raw task key']    # list[str]
        raw_task_ids = torch.tensor([self.raw_task_keys.index(key) for key in raw_task_key],
                                    dtype=torch.int,device=self.device)
        raw_input_text = batch['raw input text']  # list[str]
        all_target_texts = batch['all target texts']    # NOTE: [tuple('v0'), tuple('v1'), ...]
        prompt_ids = self.p_embedder.encode(prompts, device=self.device)  # (n, 3)
        prompt_embed = self.p_embedder(prompt_ids, self.eval_pembed)  # (n, c, 3) --> (n, c)
        
        input_ids, input_mask = self.tokenize(input_text, self.input_text_len-self.prompt_tuning_len)
        tgt_ids, _ = self.tokenize(tgt_text, self.tgt_text_len)
        sentiment_ids = self.encode_labels(sentiment_label)
        relation_ids = self.encode_labels(relation_label)

        return (eeg, eeg_mask, prompt_embed, 
                input_ids, input_mask, tgt_ids,
                prompt_ids, raw_task_ids, 
                sentiment_ids, relation_ids,
                tgt_text, raw_input_text, all_target_texts)
    
    def encode_text(self, src_ids, src_mask):
        text_encoder = self.text_model.get_encoder()  # a general method?
        with torch.no_grad():
            outputs = text_encoder(input_ids = src_ids, 
                                    attention_mask = src_mask, 
                                    return_dict = True)
        hidden_mask = src_mask

        hidden_states = outputs['last_hidden_state']
        return hidden_states, hidden_mask

    def text_decoder_forward(self, src_embeds, src_mask, tgt_ids):
        labels = tgt_ids.detach().clone()
        labels.masked_fill_(labels == self.text_model.config.pad_token_id, -100)  # in-place!
        mask = src_mask if (self.use_y_mask and self.training) else None
        outputs = self.text_model(encoder_outputs = BaseModelOutput(src_embeds), 
                                  attention_mask = mask,
                                  labels = labels)
        loss = outputs['loss']      # (1)
        logits = outputs['logits']  # (n, l, vocab_sz)
        return loss, logits.detach()
    
    def shared_forward(self, batch):
        (eeg, eeg_mask, prompt_embed, 
         input_text_ids, input_text_mask, target_text_ids, 
         prompt_ids, raw_task_ids, 
         sentiment_ids, relation_ids,
         target_text, raw_input_text, all_target_texts) = self.get_inputs(batch)
     
        input_text_embeds, hidden_text_mask = self.encode_text(input_text_ids, input_text_mask)
        
        eeg_hiddens, _ = self.eeg_encoder(eeg, eeg_mask, prompt_embed)  # TODO: return weights

        (loss_clip, logits_clip, loss_commitment, 
         eeg_embeds, eeg_emb, input_text_emb) = self.aligner(eeg_hiddens, input_text_embeds, hidden_text_mask)

        loss_lm, logits_lm = self.text_decoder_forward(eeg_embeds, hidden_text_mask, target_text_ids)
        return {'loss_commitment': loss_commitment,            # (1)
                'loss_clip': loss_clip,                        # (1)
                'loss_lm': loss_lm,                            # (1)
                'logits_clip': logits_clip,                    # (n, n)
                'logits_lm': logits_lm,                        # (n, l, vocab_size)
                'eeg_emb_vector': eeg_emb,                     # (n, e)
                'text_emb_vector': input_text_emb,             # (n, e)
                'eeg_embeds': eeg_embeds,                      # (n, l, e), for generation (w/o mask)
                # 'input_text_embeds': input_text_embeds,       # (n, l, e)
                # 'input_text_mask': input_text_mask,           # (n, l)
                ### for logging table
                'input_text_ids': input_text_ids,              # (n, l)
                'target_text_ids': target_text_ids,            # (n, l')
                'prompt_ids': prompt_ids,                      # (n, 3)
                'raw_task_ids': raw_task_ids,                  # (n)
                'sentiment_ids': sentiment_ids,    # (n)
                'relation_ids': relation_ids,      # (n)
                ### for cal metrics
                'raw_input_text': raw_input_text,              # list[str]
                'all_target_texts': all_target_texts,          # list[tuple[str]]
                }

    def define_metrics(self, metric_keys: list=None) -> None:
        run = self.logger.experiment
        for key in metric_keys:
            if 'loss' in key:
                run.define_metric(key, summary='min')
            else:
                run.define_metric(key, summary='max')

    def cal_retrieval_metrics(self, logits: torch.Tensor, targets:torch.Tensor=None,
                              strict=False):
        if strict: # NOTE only within a subset according to the `self.bsz_retrieval`
            assert logits.shape[0] >= self.bsz_retrieval
            logits = logits[:self.bsz_retrieval, :self.bsz_retrieval]  # NOTE
        bsz = logits.shape[0]
        targets = torch.arange(bsz, dtype=torch.int, 
                              device=self.device) if targets is None else targets # (n)
        probs = torch.softmax(logits, dim=-1)
        acc_top1 = multiclass_accuracy(probs, targets, average='micro', num_classes=bsz, top_k=1)
        acc_top5 = multiclass_accuracy(probs, targets, average='micro', num_classes=bsz, top_k=5)
        acc_top10 = multiclass_accuracy(probs, targets, average='micro', num_classes=bsz, top_k=10)
        return {'retrieval_acc_top01': acc_top1,
                'retrieval_acc_top05': acc_top5,
                'retrieval_acc_top10': acc_top10,
                }
    
    def training_step(self, batch, batch_idx):
        shared_outputs = self.shared_forward(batch)
        loss_commitment = shared_outputs['loss_commitment']     # (1)
        loss_clip = shared_outputs['loss_clip']                 # (1)
        loss_lm = shared_outputs['loss_lm']                     # (1)
        loss = self.λ * loss_clip + (1-self.λ) * loss_lm + self.ε * loss_commitment
        metrics = {'loss': loss,
                   'loss_commitment': loss_commitment,
                   'loss_clip': loss_clip,              
                   'loss_lm': loss_lm, 
                #    'learning_rate': self.lr_schedulers().get_last_lr()[0],
                    } 

        retrieval_metrics = self.cal_retrieval_metrics(shared_outputs['logits_clip'], strict=False)
        metrics.update(retrieval_metrics)

        metrics = {f'train/{k}': v for k, v in metrics.items()}
        if self.current_epoch == 0 and batch_idx == 0:
            self.define_metrics(list(metrics.keys()))
        self.log_dict(metrics, sync_dist=True, batch_size=self.bsz_train)
        # `self.log/self.log_dict` will also gather each metric across devices
        return loss

    def on_validation_epoch_start(self):
        if (self.current_epoch + 1) % self.full_val_interval == 0 or self.current_epoch == 0:
            # self.val_step_outputs = defaultdict(list)
            self.full_val_step_outputs = []

    def validation_step(self, batch, batch_idx):
        shared_outputs = self.shared_forward(batch)
        loss_commitment = shared_outputs['loss_commitment']     # (1)
        loss_clip = shared_outputs['loss_clip']                 # (1)
        loss_lm = shared_outputs['loss_lm']                     # (1)
        metrics = {'loss_commitment': loss_commitment,
                   'loss_clip': loss_clip,              
                   'loss_lm': loss_lm,         
                    } 

        retrieval_metrics = self.cal_retrieval_metrics(shared_outputs['logits_clip'], strict=False)  
        # NOTE: only for checkpointing here, allowing smaller batch size
        metrics.update(retrieval_metrics)
        metrics = ({f'val/{k}':v for k, v in metrics.items()})
        if self.current_epoch == 0 and batch_idx == 0:
            self.define_metrics(list(metrics.keys()))
        self.log_dict(metrics, sync_dist=True, batch_size=self.bsz_val)

        if (self.current_epoch + 1) % self.full_val_interval == 0 or self.current_epoch == 0:
            outputs = self.full_val_step(shared_outputs)
            self.full_val_step_outputs.append(outputs)
            
        
    def full_val_step(self, shared_outputs):
        bsz = shared_outputs['eeg_embeds'].shape[0]
        # prepare for gathering
        to_gather_tensors = {'eeg_emb_vector': shared_outputs['eeg_emb_vector'],    # (n, e)
                             'text_emb_vector': shared_outputs['text_emb_vector'],  # (n, e)
                             'input_text_ids': shared_outputs['input_text_ids'],    # (n, l)
                             'target_text_ids': shared_outputs['target_text_ids'],  # (n, l')
                             'prompt_ids': shared_outputs['prompt_ids'],            # (n, 3)
                             'raw_task_ids': shared_outputs['raw_task_ids'],        # (n)
                             'sentiment_ids': shared_outputs['sentiment_ids'],  # (n)
                             'relation_ids': shared_outputs['relation_ids'],  # (n)
                             }
        
        raw_input_ids, _ = self.tokenize(shared_outputs['raw_input_text'], self.input_text_len)   # (n, l)
        K = self.trainer.datamodule.n_target_text
        all_tgt_text_list = []
        for targets in zip(*shared_outputs['all_target_texts']): # iterate on each sample
            all_tgt_text_list.extend(list(targets))
        tgt_ids, _ = self.tokenize(all_tgt_text_list, self.tgt_text_len)            # (n*k, l')
        all_tgt_ids = tgt_ids.reshape(bsz, K, -1)                                   # (n, k, l')
        to_gather_tensors.update({'raw_input_text_ids': raw_input_ids,
                                  'all_target_text_ids': all_tgt_ids,
                                  })

        gen_ids_dict = self.generation_step(shared_outputs)
        to_gather_tensors.update(gen_ids_dict)
        return to_gather_tensors

    # @torch.autocast(self.device, dtype=(torch.bfloat16 if self.precision == "bf16-mixed" else torch.half))
    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.full_val_interval == 0 or self.current_epoch == 0:
            # self.val_step_outputs: dict[list[dict[str, tensor]]]
            outputs = default_collate(self.full_val_step_outputs)         # (n_steps, bsz, ...)
            outputs = {k: v.flatten(0,1) for k,v in outputs.items()}      # (n_steps*bsz, ...)
            if dist.is_initialized():
                outputs = self.all_gather(outputs)                        # (n_devices, n_steps*bsz, ...)
                outputs = {k: v.flatten(0,1) for k,v in outputs.items()}  # (n_devices*n_steps*bsz, ...)
            if self.local_rank == 0:
                with torch.autocast(device_type='cuda', dtype=(torch.bfloat16 if self.trainer.precision == "bf16-mixed" else torch.half)):
                    self.cal_and_log(outputs, prefix='full_val')
            self.full_val_step_outputs.clear()

    def cal_gen_metrics(self, pred_text: list[str], target_texts: list[tuple[str], str], 
                        raw_input_text: list[str], return_more=False) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:
        bleu1, bleu2, bleu3, bleu4 = [],[],[],[]
        rouge1_fmeasure, rouge1_precision, rouge1_recall = [],[],[]
        wer = []
        m_dicts = []
        for pred, tgts, input in zip(pred_text, target_texts, raw_input_text):
            bleu1.append(bleu_score([pred], [tgts], n_gram=1))
            bleu2.append(bleu_score([pred], [tgts], n_gram=2))
            bleu3.append(bleu_score([pred], [tgts], n_gram=3))
            bleu4.append(bleu_score([pred], [tgts], n_gram=4))
            rouge1_dict = rouge_score([pred], [tgts], rouge_keys='rouge1')
            rouge1_fmeasure.append(rouge1_dict['rouge1_fmeasure'])
            rouge1_precision.append(rouge1_dict['rouge1_precision'])
            rouge1_recall.append(rouge1_dict['rouge1_recall'])
            wer.append(word_error_rate([pred], [input]))
            if return_more:
                bleu1_mtv = {f'BLEU1@MTV{i:02d}': bleu_score([pred], [tgt], n_gram=1)
                            for i, tgt in enumerate(tgts)}
                bleu1_raw = {'BLEU1@RAW': bleu_score([pred], [input], n_gram=1)}
                bleu2_mtv = {f'BLEU2@MTV{i:02d}': bleu_score([pred], [tgt], n_gram=2) 
                            for i, tgt in enumerate(tgts)}
                bleu2_raw = {'BLEU2@RAW': bleu_score([pred], [input], n_gram=2)}
                rouge1_mtv = {f'ROUGE1@MTV{i:02d}': 
                            rouge_score([pred], [tgt], rouge_keys='rouge1')['rouge1_recall']
                            for i, tgt in enumerate(tgts)}
                rouge1_raw = {'ROUGE1@RAW': 
                            rouge_score([pred], [input], rouge_keys='rouge1')['rouge1_recall']}
                m_dicts.append({**bleu1_mtv, **bleu1_raw, **bleu2_mtv, **bleu2_raw, **rouge1_mtv, **rouge1_raw})
        # NOTE: to(self.device) is necessary before gathering when using ddp
        metrics_mean = {'bleu1': torch.stack(bleu1), 
                        'bleu2': torch.stack(bleu2), 
                        'bleu3': torch.stack(bleu3), 
                        'bleu4': torch.stack(bleu4), 
                        'rouge1_fmeasure': torch.stack(rouge1_fmeasure), 
                        'rouge1_precision': torch.stack(rouge1_precision), 
                        'rouge1_recall': torch.stack(rouge1_recall), 
                        'wer': torch.stack(wer), 
                        }
        
        return metrics_mean, m_dicts
    
    def pad_ids(self, ids):
        pad_len = self.tgt_text_len - ids.shape[1]
        if pad_len > 0:
            pad_value = self.text_model.config.pad_token_id
            ids = F.pad(ids, (0, pad_len), 'constant', pad_value)
        return ids.int()

    def convert_logits_to_ids(self, logits) -> tuple[list[str], torch.Tensor]:
        probs = logits.softmax(dim=-1)  # (bs, out_len, vocab_size)
        _, ids = probs.topk(1) # (bs, out_len, 1)
        ids = ids.squeeze().int()
        # text = self.tokenizer.batch_decode(ids, skip_special_tokens=True)  
        return ids

    def generation_step(self, shared_outputs) -> tuple[dict]:
        # generation 
        # NOTE: with batch=24, 4090D*1: num_beams=2 --> 3min15s, 4690MB; 
        #                               num_beams=4 --> 3min30s, 6924MB; 
        gen_ids = self.text_model.generate(encoder_outputs = BaseModelOutput(shared_outputs['eeg_embeds']), 
                                           num_beams = 2, 
                                           min_length = 0, max_length=self.tgt_text_len)
        out_ids_dict = {'gen_ids': self.pad_ids(gen_ids)}  # for gathering accross devices
        tf_ids = self.convert_logits_to_ids(shared_outputs['logits_lm'])
        out_ids_dict.update({'tf_ids': tf_ids})
        return out_ids_dict
    
    def cal_label_embs(self, labels: list[str], template: str=None):
        if template:
            label_sentences = [template.replace("<MASK>", label) for label in labels]
        else:
            label_sentences = labels
        ids, mask = self.tokenize(label_sentences, 32)
        embeds, _ = self.encode_text(ids, mask)
        emb_vectors = self.aligner.embed_text(embeds, mask)  # (n, e)
        return emb_vectors
    
    def run_cls(self, eeg_emb_vector, candi_emb_vector):
        eeg_norm = eeg_emb_vector / eeg_emb_vector.norm(dim=1, keepdim=True)      # (n, e)
        candi_norm = candi_emb_vector / candi_emb_vector.norm(dim=1, keepdim=True)   # (c, e)
        probs = (eeg_norm @ candi_norm.T).softmax(dim=-1)  # (n, c)
        return probs
    
    def collect_cls_preds(self, probs, target_ids, candidates: list[str], ignore_idx=-1) -> tuple[list, list[dict]]:
        n = probs.shape[0]
        c = len(candidates)
        targets, prob_dicts = [], []
        for i in range(n):
            idx = target_ids[i].item()
            target = candidates[idx] if idx !=ignore_idx else 'nan'
            targets.append(target)
            topk_probs, indices = probs[i].topk(c)
            prob_dict = {candidates[idx]: prob.item() for prob, idx in zip(topk_probs, indices)}
            prob_dicts.append(prob_dict)
        return targets, prob_dicts

    def run_sentiment_cls(self, intermediates: dict, candi_emb_vector: Tensor):
        eeg_emb_vector = intermediates['eeg_emb_vector']       
        probs = self.run_cls(eeg_emb_vector, candi_emb_vector) # (n, c)
        c = candi_emb_vector.shape[0]
        assert c == probs.shape[1] 
        target_ids = intermediates['sentiment_ids']
        acc_top1 = multiclass_accuracy(probs, target_ids, average='micro', num_classes=c, ignore_index=-1, top_k=1)
        accs = {'sentiment_cls_acc_top01': acc_top1}
        labels, prob_dicts = self.collect_cls_preds(probs, target_ids, self.sentiment_labels)
        return accs, labels, prob_dicts
    
    def run_relation_cls(self, intermediates: dict, candi_emb_vector: Tensor):
        eeg_emb_vector = intermediates['eeg_emb_vector']       
        probs = self.run_cls(eeg_emb_vector, candi_emb_vector) # (n, c)
        c = candi_emb_vector.shape[0]
        assert c == probs.shape[1] 
        target_ids = intermediates['relation_ids']
        acc_top1 = multiclass_accuracy(probs, target_ids, average='micro', num_classes=c, ignore_index=-1, top_k=1)
        acc_top3 = multiclass_accuracy(probs, target_ids, average='micro', num_classes=c, ignore_index=-1, top_k=3)
        accs = {'relation_cls_acc_top01': acc_top1,
                'relation_cls_acc_top03': acc_top3}
        labels, prob_dicts = self.collect_cls_preds(probs, target_ids, self.relation_labels)
        return accs, labels, prob_dicts
    
    def run_corpus_cls(self, intermediates: dict, candi_emb_vector: Tensor):
        eeg_emb_vector = intermediates['eeg_emb_vector']       
        probs = self.run_cls(eeg_emb_vector, candi_emb_vector) # (n, c)
        target_ids = intermediates['raw_task_ids'].detach().clone()
        target_ids.masked_fill_(target_ids==2, 1)
        acc = multiclass_accuracy(probs, target_ids, average='micro', num_classes=2, top_k=1)
        acc_dict = {'corpus_cls_acc': acc}
        return acc_dict

    def cal_and_log(self, outputs, prefix='full_val'):
        # pre-calculate embeddings of labels for classification
        se_label_embs = self.cal_label_embs(self.sentiment_labels, template="Sentiment classification: It is <MASK>.")
        re_label_embs = self.cal_label_embs(self.relation_labels, template="Relation classification: It is about <MASK>.")
        co_label_embs = self.cal_label_embs(labels=["The topic is about: movie, good or bad", 
                                                    "The topic is about: life experiences, relationship"])
        
        # cal overall classification metrics (micro)
        se_accs, se_labels, se_prob_dicts = self.run_sentiment_cls(outputs, se_label_embs)
        re_accs, re_labels, re_prob_dicts = self.run_relation_cls(outputs, re_label_embs)
        co_acc = self.run_corpus_cls(outputs, co_label_embs)
        mean_metrics = {**se_accs, **re_accs, **co_acc}
        mean_metrics = {f"{prefix}/mean_{k}":v for k,v in mean_metrics.items()}

        # cal group/sample-level meterics (generation, classification, retrieval)
        bsz = outputs['prompt_ids'].shape[0]
        p_keys = self.prompt_keys
        group_dict = defaultdict(list)
        for i in range(bsz):
            t_id, d_id, s_id = outputs['prompt_ids'][i]
            tds_key = f"{p_keys['task'][t_id]}-{p_keys['dataset'][d_id]}-{p_keys['subject'][s_id]}"
            raw_t_id = outputs['raw_task_ids'][i]
            raw_t_key = f"{self.raw_task_keys[raw_t_id]}"
            group_key = f"{tds_key}-{raw_t_key}"
            group_dict[group_key].append({k: v[i] for k,v in outputs.items()})

        all_rows = []  # list[dict] -> dataframe
        all_group_metrics = {}
        to_mean_metrics = []
        for group_key, intermediates_list_dict in sorted(group_dict.items()):
            t_key, d_key, s_key, raw_t_key = group_key.split('-')
            intermediates = default_collate(intermediates_list_dict)
            n = len(intermediates_list_dict)
            ### calculate & collect group-level metrics
            group_metrics = {}
            if self.current_epoch == 0:
                group_metrics.update({'num_samples': n})
            
            # cal generation metrics
            input_strs = self.tokenizer.batch_decode(intermediates['input_text_ids'], 
                                                     skip_special_tokens=True)
            raw_input_strs = self.tokenizer.batch_decode(intermediates['raw_input_text_ids'], 
                                                     skip_special_tokens=True)
            all_tgt_ids = intermediates['all_target_text_ids'].reshape(-1, self.tgt_text_len)
            all_tgt_strs = self.tokenizer.batch_decode(all_tgt_ids, skip_special_tokens=True)
            k = self.trainer.datamodule.n_target_text
            all_tgt_str_tuples = [tuple(all_tgt_strs[i*k:(i+1)*k]) for i in range(n)]
            gen_strs = self.tokenizer.batch_decode(intermediates['gen_ids'], skip_special_tokens=True)
            tf_strs = self.tokenizer.batch_decode(intermediates['tf_ids'], skip_special_tokens=True)
            tf_tgt_strs = self.tokenizer.batch_decode(intermediates['target_text_ids'], skip_special_tokens=True)

            m_gen, mdicts_gen = self.cal_gen_metrics(gen_strs, all_tgt_str_tuples, raw_input_strs, return_more=True)
            m_tf, _ = self.cal_gen_metrics(tf_strs, tf_tgt_strs, raw_input_strs)
            gen_metrics = {f'{k}_gen': v for k,v in m_gen.items()}
            gen_metrics.update({f'{k}_tf': v for k, v in m_tf.items()})
            group_metrics.update({k: v.mean() for k, v in gen_metrics.items()})
            
            # retrieval accs on groups (unnecessary)
            # if t_key == '<Normal Reading>': # TODO: sample this subset randomly?
            _, logits = self.aligner.align_emb_vector(intermediates['eeg_emb_vector'],
                                                        intermediates['text_emb_vector'])
            try:
                retrieval_metrics = self.cal_retrieval_metrics(logits, strict=True)
                group_metrics.update(retrieval_metrics)
            except AssertionError:  # NOTE ignore this group if bsz < self.bsz_retrieval, controled by `strict`
                pass

            # classification accs and predictions 
            se_accs, se_labels, se_prob_dicts = self.run_sentiment_cls(intermediates, se_label_embs)
            re_accs, re_labels, re_prob_dicts = self.run_relation_cls(intermediates, re_label_embs)
            co_acc = self.run_corpus_cls(intermediates, co_label_embs)
            group_metrics.update({**se_accs, **re_accs, **co_acc})

            all_group_metrics.update({f"{prefix}/{k}-{group_key}":v 
                                      for k,v in group_metrics.items()})
            
            ### collect metrics for sample-level logging
            for i in range(n):
                to_mean_metrics.append({'BLEU1@MTV': gen_metrics['bleu1_gen'][i], 
                                        'BLEU2@MTV': gen_metrics['bleu2_gen'][i], 
                                        'ROUGE1@MTV': gen_metrics['rouge1_recall_gen'][i],
                                        'ROUGE1@RAW': mdicts_gen[i]['ROUGE1@RAW'],
                                        })
                all_rows.append({
                    'LM': self.text_model_id, 'Task': t_key, 'Dataset': d_key, 'Subject': s_key, 
                    'Raw Task Key': raw_t_key, 
                    'Input Text': raw_input_strs[i], 'Target Texts': all_tgt_str_tuples[i],
                    'Generated Text': gen_strs[i], 
                    'Bleu1': gen_metrics['bleu1_gen'][i], 'Bleu2': gen_metrics['bleu2_gen'][i], 
                    'Rouge1-recall': gen_metrics['rouge1_recall_gen'][i], 
                    'GM': mdicts_gen[i],
                    'Rouge1-precision': gen_metrics['rouge1_precision_gen'][i],  
                    'Rouge1-fmeasure': gen_metrics['rouge1_fmeasure_gen'][i], 'WER': gen_metrics['wer_gen'][i],
                    'Bleu3': gen_metrics['bleu3_gen'][i], 'Bleu4': gen_metrics['bleu4_gen'][i], 
                    
                    'Sentiment label': se_labels[i], 'Sentiment Predictions': se_prob_dicts[i], 
                    'Relation label': re_labels[i], 'Relation Predictions': re_prob_dicts[i], 

                    'Target Text (Current epoch)': tf_tgt_strs[i], 'Generated Text (w/tf)': tf_strs[i], 
                    'Bleu1 (w/tf)': gen_metrics['bleu1_tf'][i], 'Bleu2 (w/tf)': gen_metrics['bleu2_tf'][i], 
                    'Rouge1-precision (w/tf)': gen_metrics['rouge1_precision_tf'][i],   
                    'Rouge1-fmeasure (w/tf)': gen_metrics['rouge1_fmeasure_tf'][i], 'WER (w/tf)': gen_metrics['wer_tf'][i],
                    'Bleu3 (w/tf)': gen_metrics['bleu3_tf'][i], 'Bleu4 (w/tf)': gen_metrics['bleu4_tf'][i], 
                    'Rouge1-recall (w/tf)': gen_metrics['rouge1_recall_tf'][i],       
                })
        to_mean_metrics = default_collate(to_mean_metrics)
        mean_metrics.update({f"{prefix}/mean_{k}":v.mean() for k,v in to_mean_metrics.items()})
        if self.current_epoch == 0:
            self.define_metrics(list(all_group_metrics.keys())+list(mean_metrics.keys()))
        # all_group_metrics = {k: v.to(self.device) for k, v in all_group_metrics.items()}
        # mean_metrics = {k: v.to(self.device) for k, v in mean_metrics.items()}
        self.log_dict(all_group_metrics, rank_zero_only=True)
        self.log_dict(mean_metrics, rank_zero_only=True)
        
        sample_metrics = pd.DataFrame(all_rows)
        self.logger.log_table(key=f'{prefix}/Samples', dataframe=sample_metrics)

    def on_test_epoch_start(self):
        assert not dist.is_initialized()  # NOTE: use single GPU to ensure the reproducibility
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):
        shared_outputs = self.shared_forward(batch)
        loss_commitment = shared_outputs['loss_commitment']     # (1)
        loss_clip = shared_outputs['loss_clip']                 # (1)
        loss_lm = shared_outputs['loss_lm']                     # (1)
        metrics = {'loss_commitment': loss_commitment,
                   'loss_clip': loss_clip,              
                   'loss_lm': loss_lm,         
                    } 
        
        retrieval_metrics = self.cal_retrieval_metrics(shared_outputs['logits_clip'], strict=True)  
        metrics.update(retrieval_metrics)
        
        metrics = ({f'test/{k}-batch{batch_idx}':v for k, v in metrics.items()})
        self.log_dict(metrics, sync_dist=True, batch_size=self.bsz_retrieval)
        self.full_val_step(shared_outputs)
        outputs = self.full_val_step(shared_outputs)
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self):
        # self.test_step_outputs: dict[list[dict[str, tensor]]]
        outputs = default_collate(self.test_step_outputs)         # (n_steps, bsz, ...)
        outputs = {k: v.flatten(0,1) for k,v in outputs.items()}      # (n_steps*bsz, ...)
        with torch.autocast(device_type='cuda', dtype=(torch.bfloat16 if self.trainer.precision == "bf16-mixed" else torch.half)):
            self.cal_and_log(outputs, prefix='test')
        self.test_step_outputs.clear()

    @torch.no_grad()
    def predict(self, eeg, eeg_mask, prompts, candidates:list[str]=["It is good.","It is bad."], generate=False):
        
        prompt_ids = self.p_embedder.encode(prompts, device=self.device)  # (n, 3)
        prompt_embed = self.p_embedder(prompt_ids, self.eval_pembed)  # (n, c, 3) --> (n, c)
        eeg_hiddens, _ = self.eeg_encoder(eeg, eeg_mask, prompt_embed)
        eeg_embs, eeg_emb_vector = self.aligner.embed_eeg(eeg_hiddens)

        label_embs = self.cal_label_embs(labels=candidates)
        probs = self.run_cls(eeg_emb_vector, label_embs)

        gen_strs = [None]*len(eeg)
        if generate:
            gen_ids = self.text_model.generate(encoder_outputs = BaseModelOutput(eeg_embs), 
                                                num_beams = 2, 
                                                min_length=0, max_length=self.tgt_text_len,
                                                )
            gen_strs = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return probs, gen_strs
    
    def predict_text_embedding(self, texts: list[str], input_template: str, candidates:list[str]):
        if input_template:
            texts = [input_template.replace("<MASK>", text) for text in texts]
        input_ids, input_mask = self.tokenize(texts, self.input_text_len-self.prompt_tuning_len)
        input_text_embeds, _ = self.encode_text(input_ids, input_mask)
        text_emb = self.aligner.embed_text(input_text_embeds, input_mask)

        label_embs = self.cal_label_embs(labels=candidates)
        probs = self.run_cls(text_emb, label_embs)
        return probs