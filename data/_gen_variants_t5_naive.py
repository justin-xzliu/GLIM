import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from rich.progress import track

class NaiveTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, prefix: str) -> None:
        texts = df['input text'].tolist()
        self.text = [prefix + text for text in texts]
    def __getitem__(self, index):
        return self.text[index]
    def __len__(self):
        return len(self.text)

def gen(model, dataset, bsz, device) -> list[str]:
    
    model = model.to(device)
    text_loader = DataLoader(dataset, batch_size=bsz,shuffle=False,drop_last=False,pin_memory=True)
    tgt_list = []
    for text_batch in track(text_loader):
        inputs = tokenizer(text_batch, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        out_ids = model.generate(input_ids=input_ids, 
                                 attention_mask=attention_mask, # Whatever set or not
                                 num_beams=4,
                                 min_length=0, max_length=64,
                                 )
        tgt = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        tgt_list += tgt
    return tgt_list



if __name__ == "__main__":
    device = "cuda:0"
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    df = pd.read_pickle('data/tmp/zuco_label_input_text.df')
    print(df.columns)

    dset1 = NaiveTextDataset(df, prefix='To English: ')
    targets1 = gen(model, dset1, 32, device) 
    df['naive rewritten'] = targets1

    dset2 = NaiveTextDataset(df, prefix='Summarize: ')
    targets2 = gen(model, dset2, 32, device) 
    df['naive simplified'] = targets2

    print(df.columns)
    pd.to_pickle(df, 'data/tmp/zuco_label_naive.df')