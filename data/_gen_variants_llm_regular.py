import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import transformers
from rich.progress import track
from datetime import datetime


class BatchDFDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.input_text = df['input text'].tolist()
        self.task = df['task'].tolist()
        self.label = df['raw label'].apply(lambda x: str(x)).tolist()
    def __getitem__(self, idx):
        return {'input text': self.input_text[idx],
                'task': self.task[idx],
                'label': self.label[idx],
                }
    def __len__(self):
        return len(self.input_text)


def create_message(input_text: str, task: str, label: str, tgt_key: str) -> list:
        p_mapping = {
            "syntax simplification": 
                ("Syntax Simplification (altering the structure of sentence to make it"
                 " easier to understand, such as using active voice, reducing clauses to phrases)."),
            "lexical simplification": 
                ("Lexical Simplification (focusing on the choice of words used in the sentence,"
                 " such as using simpler and more common words, avoiding jargon and technical terms)."),
            "semantic clarity": 
                ("Semantic Clarity (ensuring the meaning of sentence is clear and unambiguous,"
                 " such as limiting the use of pronouns, completing the missing subject or object)."),
        }
        principle = p_mapping[tgt_key]
        # Define the general prompt structure based on tasks
        sys_content = (f"You are an English language expert, and your task is to simplify English sentence"
                       f" according to the principle of {principle}\n\nRequirements:\n"
                       f"1. just write one simplified sentence and do not output any other texts;\n"
                       f"2. retain the sentence's core information according to the 'Hint' if they are relevant,"
                       f" otherwise, output an '<ERROR>' and explain the reason."
                       )
        
        if task == 'task1':
            hint = ("Hint: the below sentence is from a movie review and thus the opinion especially the"
                    " sentiment is the core information.")
                        
        elif task == 'task3' and label != 'nan':
            hint = (f"Hint: the below sentence is a biographical sentence about notable people and"
                    f" the relation about {label} the core information.")

        else:
            hint = ("Hint: the below sentence is a biographical sentence about notable people and"
                    " the relations, such as "
                    " 'awarding', 'education', 'employment', 'foundation', 'job title', 'nationality', "
                    " 'political affiliation', 'visit' and 'marriage' are what we are most concerned about.")
            
        message = [{"role": "system", "content": sys_content}, 
                   {"role": "user", "content": f"{hint}\nSentence: {input_text}"}]    
        return message


def gen(pipe, dataset: Dataset, tgt_key, bsz=16, k=2) -> list[str]:
    
    loader = DataLoader(dataset, batch_size=bsz, shuffle=False, drop_last=False)
    tgt_rows = []  # list[dict]
    for batch in track(loader):
        prompts = []
        for i in range(bsz):
            message = create_message(batch['input text'][i], batch['task'][i], batch['label'][i], tgt_key)
            inputs = pipe.tokenizer.apply_chat_template(message, 
                                                        tokenize=False, 
                                                        add_generation_prompt=True)
            prompts.append(inputs)
        
        outputs = pipe(prompts,
                       batch_size = bsz,
                       max_new_tokens = 64,
                       num_return_sequences=k,
                       )
        for prompt, output in zip(prompts, outputs):
            tgts = {f'{tgt_key} (v{i})': output[i]['generated_text'][len(prompt):] for i in range(k)}
            tgt_rows.append(tgts)
    df = pd.DataFrame(tgt_rows)
    return df


if __name__ == "__main__":
    start_time = datetime.now()
    df = pd.read_pickle('./data/tmp/zuco_label_input_text.df')
    dataset = BatchDFDataset(df)

    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    pipe = transformers.pipeline(
        "text-generation",
        model_id,
        device_map = "auto",
        torch_dtype=torch.float16,
        # model_kwargs = {
        #     "quantization_config": transformers.BitsAndBytesConfig(load_in_4bit=True,
        #                                                         bnb_4bit_compute_dtype=torch.float16),},
        )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side='left'

    bsz = 16
    k=2
    
    df1 = gen(pipe, dataset, 'syntax simplification', bsz, k) 
    pd.to_pickle(df1, './data/tmp/zuco_label_syntax_simplification.df')

    df2 = gen(pipe, dataset, 'lexical simplification', bsz, k) 
    pd.to_pickle(df2, './data/tmp/zuco_label_lexical_simplification.df')

    df3 = gen(pipe, dataset, 'semantic clarity', bsz, k) 
    pd.to_pickle(df3, './data/tmp/zuco_label_semantic_clarity.df')

    elapsed = datetime.now() - start_time
    print(f"Done!😀 Took: {elapsed}")