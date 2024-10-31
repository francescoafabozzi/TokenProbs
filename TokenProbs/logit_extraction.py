import os
#LD_PATH = os.getenv('LD_LIBRARY_PATH', '')
#os.environ['LD_LIBRARY_PATH'] = LD_PATH + ':/usr/local/cuda-12.3'
#os.environ['BNB_CUDA_VERSION']='123'
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.nn.functional import softmax
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from transformers import BitsAndBytesConfig
from torch.utils.data import Dataset,DataLoader
from accelerate import Accelerator
from peft import PeftModel


'''
Linux: 
conda create -n tokenprobs python=3.12
conda activate tokenprobs

Windows:
conda create -n tokenprobs python=3.12
conda activate tokenprobs

Issues with CUDA / torch:
1. Identify the CUDA version:
    nvcc --version
2. Install the matching torch version (example below for CUDA 12.3):
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu123
'''

class LogitExtractor:
    def __init__(self,model_name,adapter_name=None,quantization=None,load_model=True):
        '''
        Parameters
        ----------
        model_name : str
            Name or path of the pre-trained (generative) model to load
        adapter_name : str, optional
            Name or path of a PEFT adapter to load. Default is None 
        quantization : str or BitsAndBytesConfig, optional
            Quantization mode - either '4bit', '8bit' or None. Default is None (full precision).
                If a BitsAndBytesConfig object is provided, it will be used as is.
        load_model : bool, optional
            Whether to load the model during initialization. Default is True.
        '''
        
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.quantization = quantization

        if load_model:
            self.load_model(quantization)
            self.load_adapter()
        
    def load_model(self,quantization=None):
        '''
        Load the model with the specified quantization mode.
        '''
        
        if (type(quantization) == str) or (quantization is None):
            quantization_config = {
                '4bit': BitsAndBytesConfig(load_in_4bit=True),
                '8bit': BitsAndBytesConfig(load_in_8bit=True),
                None: None
            }[quantization]
        elif type(quantization) == BitsAndBytesConfig:
            quantization_config = quantization
        else:
            raise ValueError(f"Invalid quantization mode: {quantization}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config = quantization_config,
            device_map = 'auto',
            trust_remote_code=True
        )
        #self.model.config.use_cache = False
        #self.model.config.pretrained_tp = 1
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        
    def load_adapter(self):
        '''
        Load a PEFT adapter if specified.
        '''
        if self.adapter_name is None:
            pass
        else:
            self.model = PeftModel.from_pretrained(self.model,self.adapter_name)
            
    def get_dataloader(self,input_text,batch_size=1):
        '''
        Get a DataLoader for the input text.
        
        Parameters
        ----------
        input_text : list of str
            List of input text strings
        batch_size : int, optional
            Batch size. Default is 1
        '''
        
        ds = GenerativeDataset(input_text,self.tokenizer)
        collator = DataCollatorWithPadding(self.tokenizer)
        dataloader = DataLoader(ds,batch_size=batch_size,collate_fn=collator)
        
        return dataloader
    
    def identify_tokens(self,class_tokens = None,example_input=None):

        if not hasattr(self,'tokenizer'):
            raise ValueError("Model must be loaded first `load_model`.")

        if type(class_tokens) == dict:
            self.token_dict = class_tokens
        
        elif type(class_tokens[0]) == str:
            self.token_dict = {
                i: self.tokenizer(i)['input_ids'][1] for i in class_tokens
            }

        elif type(class_tokens[0]) == int:
            self.token_dict = {
                self.tokenizer.decode(i): i for i in class_tokens
            }
        
        else:
            raise ValueError('class_tokens must be a list of TokenIDs or the string-based tokens themselves.')

        print(f'TokenID mappings:\n\t {self.token_dict}')
        return self.token_dict
    
    def identify_tokens_new(self,class_tokens,example_input):
        '''
        Identify the token IDs for the specified tokens.
        
        Parameters
        ----------
        class_tokens : list of int or list of str or dict with token strings as keys and token IDs as values
            List of token IDs or token strings to identify.
            
        Returns
        -------
        dict with token strings as keys and tokenIDs as values
        '''

        if not hasattr(self,'tokenizer'):
            raise ValueError("Model must be loaded first `load_model`.")

        if type(class_tokens[0]) == str:
            self.token_dict = {
                i: self.tokenizer(i)['input_ids'][1] for i in class_tokens
            }
        else: 
            raise ValueError('class_tokens must be a list of tokens (str)')
        
        token_dict = {}
        output_dict = {}
        for i in class_tokens:
            token_dict[i] = self.tokenizer(example_input + i)['input_ids'][-1]
            output_dict[i] = example_input + self.tokenizer.decode(token_dict[i])
            
        print('\n\n')
        print(f'TokenID mappings:\n\t {token_dict}')
        print('--------------------------------------\n\n')
        print('Based on the prompt (for first example) + TokenIDs input. This is an example response for each class:')
        for i in class_tokens:
            print(f'Class: {i}:')
            print(f'\t{i}: {output_dict[i]}\n')
        
        print('Here are the mappings from token string to TokenIDs:')
        print(token_dict)
        self.token_dict = token_dict
        return token_dict
    
  
        
    def logit_extraction(self,input_data,tokenIDs,batch_size=1):
        '''
        Extract logits for the specified tokens from the model.
        
        Parameters
        ----------
        input_data : list of str or DataLoader
            Input data to extract logits from
        tokenIDs : list of int or list of str
            List of token IDs to extract logits for or token strings to identify the tokens by
        batch_size : int, optional
            Batch size. Default is 1
        '''
        example_input = input_data[0] if type(input_data) == list else input_data
        
        # Load input data
        if type(input_data) == list:
            input_data = self.get_dataloader(input_data,batch_size=batch_size)
        elif type(input_data) == str:
            input_data = self.get_dataloader([input_data],batch_size=batch_size)
        #elif type(input_data) == DataLoader:
        #    pass
        else:
            raise ValueError('Invalid input_data type')
        
        self.token_dict = self.identify_tokens(tokenIDs,example_input)
        
        # Set device (GPU or CPU)
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        
        # Prepare model and input data for GPU/CPU
        accelerator = Accelerator()
        self.model,input_data = accelerator.prepare(self.model,input_data)
        
        left_pad = True if self.tokenizer.padding_side == 'left' else False
        # Get predictions
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in tqdm(input_data):
                output = self.model(
                    input_ids = batch['input_ids'].to(device),
                    attention_mask = batch['attention_mask'].to(device)
                ).logits.detach().cpu()
                
                # Extract logits following the end of the prompt tokens
                if left_pad: 
                    generated = output[:,-1,:]
                else:
                    generated = torch.vstack(
                        [output[i,batch['length'].to('cpu')[i]-1,:] for i in range(len(batch['length'].to('cpu')))]
                )
                
                # Extract logits for the specified tokens
                preds.append(
                    softmax(generated[:,list(self.token_dict.values())],dim=-1).numpy()
                )
                
                torch.cuda.empty_cache()
                
        output_df = pd.DataFrame(np.vstack(preds),columns=list(self.token_dict.keys()))
        
        return output_df
    
    
    def text_generation(self,input_data,batch_size=1,max_new_tokens=100):
        '''
        Generate text from the model.
        
        Parameters
        ----------
        input_data : list of str or DataLoader
            Input data to generate text from
        batch_size : int, optional
            Batch size. Default is 1
        max_new_tokens : int, optional
            Maximum number of new tokens to generate. Default is 100
        '''
        
        self.tokenizer.padding_side = "left"
        
        if type(input_data) == list:
            input_data = self.get_dataloader(input_data,batch_size=batch_size)
        elif type(input_data) == str:
            input_data = self.get_dataloader([input_data],batch_size=batch_size)
        elif type(input_data) == DataLoader:
            pass
        else:
            raise ValueError('Invalid input_data type')
        
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
        # Prepare model and input data for GPU/CPU
        accelerator = Accelerator()
        self.model,input_data = accelerator.prepare(self.model,input_data)
        
        # Generate text
        preds = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(input_data):
                output = self.model.generate(
                    input_ids = batch['input_ids'].to(device),
                    attention_mask = batch['attention_mask'].to(device),
                    max_new_tokens = max_new_tokens,
                    eos_token_id = self.tokenizer.eos_token_id
                )
                
                preds.extend(self.tokenizer.batch_decode(output,skip_special_tokens=True))

                torch.cuda.empty_cache()
                
        return preds
            
    
class GenerativeDataset(Dataset):

    def __init__(self,data,tokenizer,train=False):
        if type(data) == list:
            self.prompts = data

        elif type(data) == pd.DataFrame:
            self.df = data.copy()
            self.prompts = data.prompt.tolist()
            
        else:
            raise ValueError('Input data must be a list of prompted text inputs or a dataframe with a column `prompt`.')
        
        self.tokenizer = tokenizer 
        self.train = train

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self,idx):
        
        prompts = self.prompts[idx]
        encoded = self.tokenizer(
            prompts,return_length=bool(1-self.train),
            truncation=True,padding=False
        )
        return encoded
        
'''        
model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
peft_name = None
quantization = '8bit'


le2 = LogitExtractor(model_name,peft_name,quantization,load_model=True)

from datasets import load_dataset
def download_financial_phrasebank():
    data = load_dataset('financial_phrasebank','sentences_50agree')
    data = data['train'].train_test_split(seed = 42)
    train_df = data['train'].to_pandas()
    test_df = data['test'].to_pandas()
    
    return {
        'train':train_df,
        'test':test_df
    }
    
    
le2.tokenizer.padding_side = "right"
new_id_output_right = le2.logit_extraction(eval_text,['negative','neutral','positive'],batch_size=16)
new_id_preds_right = new_id_output_right.idxmax(axis=1)

old_id_output_left = le2.logit_extraction(eval_text,['negative','neutral','positive'],batch_size=16)
old_id_preds_left = old_id_output_left.idxmax(axis=1)

old_id_output_right = le2.logit_extraction(eval_text,['negative','neutral','positive'],batch_size=16)
old_id_preds_right = old_id_output_right.idxmax(axis=1)

#new_id_acc_left = (new_id_preds == eval_labels).sum()/len(eval_labels) # new_id_output
#new_id_acc_right = (new_id_preds_right == eval_labels).sum()/len(eval_labels) # new_id_output_right
old_id_acc_left = (old_id_preds_left == eval_labels).sum()/len(eval_labels) # old_id_output_left
old_id_acc_right = (old_id_preds_right == eval_labels).sum()/len(eval_labels) # old_id_output_right

data = download_financial_phrasebank()
train_df, test_df = data['train'], data['test']

def sentiment_prompt(input_text):
    intructions = "### Instructions: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    news_input = f"### Input: {input_text}"
    response_format = "### Response: "
    return f"{intructions}\n{news_input}\n{response_format}"

def convert_labels(label):
    return {
        0:"negative",
        1:'neutral',
        2:'positive',
    }[label]


eval_text = test_df.sentence.apply(sentiment_prompt).tolist()
eval_labels = test_df.label.apply(convert_labels).tolist()

#le2.tokenizer.padding_side = "right"
le2.tokenizer.padding_side = "left"
fpb_output2 = le2.logit_extraction(eval_text,['negative','neutral','positive'],batch_size=16)
fpb_preds2 = fpb_output2.idxmax(axis=1)

eval_df = pd.DataFrame({
    'pred':fpb_preds,
    'label':eval_labels
})


eval_df2 = pd.DataFrame({
    'pred':fpb_preds2,
    'label':eval_labels
})


# With old token identification (right padding)
eval_df3 = pd.DataFrame({
    'pred':fpb_preds3,
    'label':eval_labels
})

# With old token identification (left padding)
eval_df4 = pd.DataFrame({
    'pred':fpb_preds4,
    'label':eval_labels
})

(eval_df.pred == eval_df.label).sum()/len(eval_df) # Right padding with new token identification
(eval_df2.pred == eval_df2.label).sum()/len(eval_df2) # Left padding with new token identification
(eval_df3.pred == eval_df3.label).sum()/len(eval_df3) # Right padding with old token identification
(eval_df4.pred == eval_df4.label).sum()/len(eval_df4) # Left padding with new token identification


dl = le2.get_dataloader(eval_text,batch_size=16)

output = []
for batch in dl:
    output.append(le2.model(input_ids = batch['input_ids'].to('cuda'),
    attention_mask = batch['attention_mask'].to('cuda')
    ).logits.detach().cpu())
    
    
output[0].shape
softout = softmax(output[0][:,-1,list(le2.token_dict.values())])
softout2 = softmax(output[0][:,-2,list(le2.token_dict.values())])
softout



np.argmax(output[0])
np.argmax(softout,axis=-1)
np.argmax(softout2,axis=-1)

softout2


fpb_preds
fpb_preds2

tt = pd.DataFrame(softout2,columns=list(le2.token_dict.keys()))
alt = tt.idxmax(axis=1)
alt
eval_labels
fpb_preds
pd.DataFrame(softout)

fpb_output2.iloc[0]
le2.tokenizer.decode(15)
le2.tokenizer.decode(8389)
le2.token_dict

'''