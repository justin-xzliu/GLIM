import os
import torch
import numpy as np
import pandas as pd
import lightning as pl
import torch.distributed as dist
from typing import Literal, Iterator, Union
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class GLIMDataModule(pl.LightningDataModule):
    
    def __init__(self, 
                 data_path: os.PathLike,
                 eval_noise_input: bool = False,
                 bsz_train = 64,
                 bsz_val = 24,
                 bsz_test = 24,
                 test_set_key: Literal['test', 'train', 'val'] = 'test',
                 num_workers: int = 0,
                 ):
        super().__init__()
        assert os.path.exists(data_path)
        self.data_path = data_path
        self.eval_noise_input = eval_noise_input
        self.bsz_train = bsz_train
        self.bsz_val = bsz_val  
        self.bsz_test = bsz_test
        self.test_set_key = test_set_key
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        try:
            local_rank = os.environ["LOCAL_RANK"]
        except KeyError:
            local_rank = '0'
        print(f'[Rank {local_rank}][{self.__class__.__name__}] running `setup()`...', end='\n')
        df = pd.read_pickle(self.data_path)
        if stage == "fit":
            self.train_set = ZuCoDataset(df, 'train')
            self.val_set = ZuCoDataset(df, 'val', self.eval_noise_input)
            self.n_target_text = self.val_set.n_target_text
        elif stage == "test":
            self.test_set = ZuCoDataset(df, 'test', self.eval_noise_input)
            self.n_target_text = self.test_set.n_target_text
        print(f'[Rank {local_rank}][{self.__class__.__name__}] running `setup()`...Done!','\U0001F60B'*3)
            
    def train_dataloader(self):
        train_sampler = GLIMSampler(self.train_set, self.train_set.data['text uid'],
                                    'train', self.bsz_train)
        train_loader = DataLoader(self.train_set,
                                  batch_sampler=train_sampler,
                                  num_workers = self.num_workers,
                                  pin_memory=True,
                                  )
        return train_loader

    def val_dataloader(self):
        val_sampler = GLIMSampler(self.val_set, self.val_set.data['text uid'], 
                                  'val', self.bsz_val)
        val_loader = DataLoader(self.val_set,
                                batch_sampler = val_sampler,
                                num_workers = self.num_workers,
                                pin_memory=True,
                                )
        return val_loader
    
    def test_dataloader(self):
        test_sampler = GLIMSampler(self.test_set, self.test_set.data['text uid'], 
                                   'test', self.bsz_test)
        test_loader = DataLoader(self.test_set,
                                 batch_sampler = test_sampler,
                                 num_workers = self.num_workers,
                                 pin_memory=True,
                                 )
        return test_loader


class GLIMSampler(DistributedSampler):
    '''
    A batch sampler for train/val/test GLIM on `ZuCo1` + `ZuCo2`.  
    It samples batches by the `text` rather than `eeg-text pair` to make sure the `clip loss` works properly
    '''
    def __init__(self, 
                 dataset: Dataset, 
                 identifiers: list,
                 phase: Literal['train', 'val', 'test'],
                 batch_size: int,
                 num_replicas = None,
                 rank = None,
                 ) -> None:
        if (num_replicas is None) and (not dist.is_initialized()):
            self.dataset = dataset
            self.num_replicas = 1
            self.rank = 0
            self.epoch = 0
            self.seed = 0
        else:
            super().__init__(dataset, num_replicas=num_replicas, rank=rank)
            # set 4 attributes inside: self.dataset, self.num_replicas, self.rank, self.epoch, self.seed
            del self.num_samples
            del self.total_size
        self.shuffle, self.drop_last = (True, True) if phase == 'train' else (False, True) 
        # NOTE: drop_last is inevitable, see `sample_batches()`
        self.phase = phase
        self.batch_size = batch_size
        self.identifiers = torch.tensor(identifiers)
        
        # uni_text_uids, counts = torch.unique(self.text_ids, return_counts=True, dim=0)
        # n_uni_text = len(uni_text_uids)
        # assert n_uni_text // (self.num_replicas * self.batch_size) > 0
        self.n_batches_per_device = self.estimate_len()
        self.n_batches = self.n_batches_per_device * self.num_replicas

    def __len__(self) -> int:
        return self.n_batches_per_device

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(len(self.dataset))

        batches, _ = self.sample_batches(indices,
                                        identifiers = self.identifiers[indices], 
                                        batch_size = self.batch_size,
                                        )
        if len(batches) >= self.n_batches:
            batches = batches[:self.n_batches]
        else:
            padding_size = self.n_batches - len(batches)
            if padding_size > 3:
                print('😅😅😅 [padding_size>3]!!! ', f'expect {self.n_batches} batches but only got {len(batches)}...')
                print('epoch:             ',f'{self.epoch}')
                print('phase:             ',f'{self.phase}')
                print('batch_size:        ',f'{self.batch_size}')
            batches += batches[:padding_size]
        sub_batches = batches[self.rank : self.n_batches : self.num_replicas]

        for batch in sub_batches:
            yield batch

    def estimate_len(self, k=10):
        if self.shuffle:
            batch_nums = []
            for i in range(k):
                g = torch.Generator()
                g.manual_seed(self.seed + i)  # for reproducibility
                indices = torch.randperm(len(self.dataset), generator=g)
                batches, _ = self.sample_batches(indices,
                                                identifiers = self.identifiers[indices], 
                                                batch_size = self.batch_size,
                                                )
                batch_nums.append(len(batches))
            batch_num = min(batch_nums)
        else:
            indices = torch.arange(len(self.dataset))
            batches, _ = self.sample_batches(indices,
                                            identifiers = self.identifiers[indices], 
                                            batch_size = self.batch_size,
                                            )
            batch_num = len(batches)
        estimated_len = batch_num // self.num_replicas
        return estimated_len
    
    @torch.no_grad()
    def sample_batches(self, indices, identifiers, batch_size, exhaust = False):
        '''
        Sampling batches by `identifiers` rather than `indices` of samples makes sure that 
        all samples within a batch have distinct identifiers.

        Inputs:
            `indices`:      torch.tensor, (n), sample indices ranging from 0 to n-1
            `indentifiers`: torch.tensor, (n), `text uids` with the same order coressponding to `indices`
        '''
        # Group samples by `task id`
        ids_idents = torch.stack([indices, identifiers],dim=1)
        batches = []
        
        unused_ids_idents = ids_idents
        n_uni_idents = torch.unique(ids_idents[:,1]).shape[0]
        # print('n_uni_idents:      ',f'{n_uni_idents}')
        while n_uni_idents >= batch_size:
            valid_batches, unused_ids_idents = self.non_overlapping_sample(unused_ids_idents, self.batch_size)
            batches.extend(valid_batches)
            if len(unused_ids_idents) == 0:
                n_uni_idents = 0
                break
            n_uni_idents = torch.unique(unused_ids_idents[:,1]).shape[0]  
        # batches: list[Tensor(bs, 2)]
        batches_ids = [batch[:,0].int().tolist() for batch in batches] # list[list[int]]

        exhausted_batches = []
        if exhaust and len(unused_ids_idents) != 0: 
            while n_uni_idents > 0:
                valid_batches, unused_ids_idents = self.non_overlapping_sample(unused_ids_idents, n_uni_idents)
                exhausted_batches.extend(valid_batches)
                if len(unused_ids_idents) == 0:
                    break
                n_uni_idents = torch.unique(unused_ids_idents[:,1]).shape[0] 
            # exhausted_batches = list[Tensor(bs*, 2)]
            exhausted_batches_ids = [batch[:,0].int().tolist() for batch in exhausted_batches] # list[list[int]]
        else:
            exhausted_batches_ids = []
        return batches_ids, exhausted_batches_ids
    
    def non_overlapping_sample(self, ids_idents: torch.Tensor, 
                               batch_size: int) -> tuple[list[torch.Tensor], Union[torch.Tensor, list]]:
        valid_batches = []
        used_idents = set()
        current_batch = []
        unused_samples = []
        for idx_ident in ids_idents:
            idx, ident = idx_ident
            if ident.item() not in used_idents: # Track identifiers used in the current batch to ensure uniqueness
                used_idents.add(ident.item())
                current_batch.append(idx_ident)
            else:
                unused_samples.append(idx_ident)

            if len(current_batch) == batch_size:
                valid_batches.append(torch.stack(current_batch))
                current_batch = []
                used_idents = set()
        # Include the last batch if it has fewer than batch_size elements
        if current_batch:
            unused_samples.extend(current_batch)
        try:
            unused_samples = torch.stack(unused_samples)
        except:
            unused_samples = []
        return (valid_batches,   # list[tensor(bs, 2)]
                unused_samples,  # tensor(n, 2)
                )


class ZuCoDataset(Dataset):

    def __init__(self, 
                 df: pd.DataFrame,
                 phase: Literal['train', 'val', 'test'],
                 eval_noise_input: bool = False,
                 ):
        # pt_target_keys = ['input text']
        pt_target_keys = ['lexical simplification (v0)', 'lexical simplification (v1)', 
                          'semantic clarity (v0)', 'semantic clarity (v1)', 
                          'syntax simplification (v0)', 'syntax simplification (v1)',
                          'naive rewritten', 'naive simplified']
        df = df[df['phase'] == phase]
        if phase == 'train':
            target_keys = pt_target_keys
            data_dicts = []
            for target_key in target_keys:
                data = self.__fetch_from_df(df, target_key)
                data_dicts.append(data)
            data = collate_fn(data_dicts)
            targets_tuple_list = [(text, ) for text in data['target text']]
        else:
            data = self.__fetch_from_df(df, "input text")
            if eval_noise_input:
                data['eeg']
            target_lists = [df[key].values.tolist() for key in pt_target_keys]
            targets_tuple_list = list(zip(*target_lists))
        data.update({"all target texts": targets_tuple_list})

        if eval_noise_input:
            n = len(data['eeg'])
            l,c = data['eeg'][0].shape
            data.pop('eeg')
            data.pop('mask')
            rng = np.random.default_rng(seed=42)
            data['eeg'] = rng.standard_normal((n,l,c), dtype=np.float32)
            data['mask'] = np.ones((n,l), dtype=np.int8)

        self.n_target_text = len(pt_target_keys)
        self.data = data
        
    def __fetch_from_df(self, df, target_key):
        

        input_template = "To English: <MASK>"
        raw_input_text = df['input text'].tolist()
        input_text = [input_template.replace("<MASK>", src) for src in raw_input_text]
        target_text = df[target_key].tolist()
        
        raw_t_keys = df['task'].tolist()
        t_prompts = ['<NR>' if t_key != 'task3' else '<TSR>' for t_key in raw_t_keys]
        # t_prompts = ['<NR>'] * len(raw_t_keys)
        d_prompts = df['dataset'].tolist()
        s_prompts = df['subject'].tolist()
        prompt = list(zip(t_prompts, d_prompts, s_prompts))
        text_uid = df['text uid'].values.tolist()

        sentiment_label  = df['sentiment label'].apply(lambda x: str(x)).values.tolist()
        relation_label = df['relation label'].apply(lambda x: str(x)).values.tolist()
        eeg = df['eeg'].tolist()
        mask = df['mask'].tolist()
        return {'eeg': eeg,                   # list[np.arrary], [(l, c),]
                'mask': mask,                 # list[np.arrary], [(l),], 1 for unmasked; 0 for masked
                'prompt': prompt,             # list[tuple[str]], [('task', 'dataset', 'subject')]
                'text uid': text_uid,         # list[int]
                'input text': input_text,     # str
                'target text': target_text,   # str
                'sentiment label': sentiment_label,                         # str
                'relation label': relation_label,                           # str
                'raw task key': raw_t_keys,                                 # str
                'raw input text': raw_input_text,                           # str
                }

    def __len__(self):
        return len(self.data['eeg'])
    
    def __getitem__(self, idx):
        return {
                'eeg': torch.from_numpy(self.data['eeg'][idx]),       # tensor, float32, (*, l, c)
                'mask': torch.from_numpy(self.data['mask'][idx]),     # tensor, int8, (*, l), 1 for unmasked; 0 for masked
                'prompt': self.data['prompt'][idx],             # tuple[str], [('task', 'dataset', 'subject')]
                'text uid': self.data['text uid'][idx],         # int
                'input text': self.data['input text'][idx],     # str
                'target text': self.data['target text'][idx],   # str
                'sentiment label': self.data['sentiment label'][idx], # str
                'relation label': self.data['relation label'][idx],   # str
                'raw task key': self.data['raw task key'][idx],         # str
                'raw input text': self.data['raw input text'][idx],     # str
                'all target texts': self.data['all target texts'][idx],   # tuple(str)
                }


def collate_fn(batch_list: list[dict]) -> dict:
    collated_batch = {}
    for k, v in batch_list[0].items():
        if isinstance(v, torch.Tensor):
            collated_batch[k] = torch.cat([batch[k] for batch in batch_list])
        else:
            collated_batch[k] = []
            for batch in batch_list:
                collated_batch[k].extend(batch[k])
    return collated_batch
