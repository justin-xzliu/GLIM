import pandas as pd
import torch
from torch.utils.data import DataLoader
import transformers
from rich.progress import track
from datetime import datetime


def gen(pipe, texts: list, tgt_key='simplified text', bsz=16, k=8) -> list[str]:
    if tgt_key == 'simplified text':
        prefix = {"role": "system", 
            "content": 
                ("Your task is to simplify the following sentence as much as possible."
                " Please retain its core information and just output the simplified sentence."
                )}
    elif tgt_key == 'rewritten text':
        prefix = {"role": "system", 
            "content": 
                ("Your task is to rewrite the following sentence. Please retain its core information and"
                 " just output the rewritten sentence."
                )}
    else:
        raise KeyError  
    # generate multiple samples at a time!
    
    text_loader = DataLoader(texts, batch_size=bsz, shuffle=False, drop_last=False)
    tgt_rows = []  # list[dict]
    for batch in track(text_loader):
        messages = [[prefix, {"role": "user", "content": text}] for text in batch]
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        pipe.tokenizer.padding_side='left'
        prompts = [pipe.tokenizer.apply_chat_template(message, 
                                                            tokenize=False, 
                                                            add_generation_prompt=True) 
                                                            for message in messages]
        outputs = pipe(prompts,
                        batch_size = bsz,
                        max_new_tokens=64,
                        eos_token_id=[pipe.tokenizer.eos_token_id, 
                                    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                        pad_token_id = pipe.tokenizer.eos_token_id,
                        num_return_sequences=k,
                        )
        for prompt, output in zip(prompts, outputs):
            tgts = {f'{tgt_key} (v{i})': output[i]['generated_text'][len(prompt):] for i in range(k)}
            tgt_rows.append(tgts)
    df = pd.DataFrame(tgt_rows)
    return df



if __name__ == "__main__":
    start_time = datetime.now()

    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model_id,
        model_kwargs = {
            "torch_dtype": torch.float16,
            "quantization_config": transformers.BitsAndBytesConfig(load_in_4bit=True,
                                                                bnb_4bit_compute_dtype=torch.float16),},
        device_map = "auto",
        )

    df = pd.read_pickle('./data/tmp/zuco_label_input_text.df')
    input_text = df['input text'].values.tolist()

    df_simplified = gen(pipeline, input_text, tgt_key='simplified text', bsz=16, k=8) 
    pd.to_pickle(df_simplified, './data/tmp/zuco_simplified_text.df')

    df_rewritten = gen(pipeline, input_text, tgt_key='rewritten text', bsz=16, k=8) 
    pd.to_pickle(df_rewritten, './data/tmp/zuco_rewritten_text.df')

    elapsed = datetime.now() - start_time
    print(f"Done!😀 Took: {elapsed}")