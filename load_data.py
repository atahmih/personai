import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset 
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def load_data():
    personachat = load_dataset('AlekseyKorshuk/persona-chat')
    # print(personachat)

    # Get a sample from the training set
    sample = personachat['train'][0]
    # print(f'sample structure: {sample.keys()}, \n persona: {sample['personality']}, \n utterances: {sample['utterances']}')
    print(f'sample structure: {sample.keys()}')
    return personachat


# split the utterances into human and bot responses
def split_utterances(data_split):
    data = []

    for sample in tqdm(data_split):
        # sample = data_split[i]
        persona = sample['personality']
        utterances = sample['utterances']
        # print(persona, utterances)

        # concatenate a persona text
        persona_text = ''.join(persona)

        # conversation turns - starts from human's first utterance
        for i in range(1, len(utterances), 2):
            if i >= len(utterances):
                break
            
            human_utterance = utterances[i-1]
            bot_response = utterances[i]

            # print(human_utterance, bot_response)

            data.append({
                'persona': persona_text,
                'human_utterance': human_utterance,
                'bot_response': bot_response
            })
    # pd.DataFrame(data).to_csv('persona_chat.csv', index=False)
    return pd.DataFrame(data)

def tokenize_data(df, tokenizer, max_length=1024):
    input_texts = []
    output_texts = []

    for _, row in df.iterrows():
        input_text = f'[Persona] {row['persona']} [Human] {row['human_utterance']} [AI]'
        output_text = str(row['bot_response'])

        input_texts.append(input_text)
        output_texts.append(output_text)

    print(type(input_texts[0]), type(output_texts[0]))
    inputs = tokenizer(input_texts, padding='max_length', truncation=False, 
                        max_length=max_length, return_tensors='pt')
    
    # tokenizer behaves differently on target, hence setting target mode
    with tokenizer.as_target_tokenizer():
        outputs = tokenizer(output_texts, padding='max_length', truncation=False, 
                            max_length=max_length, return_tensors='pt')
    return {
        'input_ids': inputs,
        'attention_mask': inputs.attention_mask,
        'labels': outputs.input_ids
    }

class PersonaDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.labels = tokenized_data['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getiem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

def setup_model():
    model = AutoModelForCausaLM.from_pretrained(
        'llama',
        torch_dtype=torch.float16 ,
        device_map=None
    )

    # Using LoRA for efficient fine-tuning
    target_modules = ['q_proj', 'v_proj']

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias='none',
        target_models=target_modules
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

return model, tokenizer

def training(model, tokenizer, train_data, val_data, )


persona_chat = load_data()
# print(persona_chat)
train_data = split_utterances(persona_chat['train'])
val_data = split_utterances(persona_chat['validation'])

print(f'{len(train_data)} training samples, {len(val_data)} testing samples')

tokenized_train = tokenize_data(train_data, tokenizer)
tokenized_val = tokenize_data(val_data, tokenizer)

print(type(tokenized_train['input_ids']))

train_dataset = PersonaDataset(tokenized_train)
val_dataset = PersonaDataset(tokenized_val)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

