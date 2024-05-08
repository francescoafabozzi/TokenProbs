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
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    TrainingArguments
)
from torch.utils.data import Dataset,DataLoader
from accelerate import Accelerator
from peft import LoraConfig
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
from peft import PeftModel


class LogitExtractor:

    def __init__(self,model_name,adapter_name=None,quantization=None):
        '''
        model_name: str
        '''

        self.model_name = model_name
        self.adapter_name = adapter_name
        self.load_model(quantization)
        self.load_adapter()


    def load_model(self,quantization=None):
        if quantization == '8bit':
            #load_in_8bit = True
            #load_in_4bit = False
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == '4bit':
            #load_in_4bit = True
            #load_in_8bit = False
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        else:
            load_in_4bit = False
            load_in_8bit=False
            quantization_config = None

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            #load_in_8bit= load_in_8bit,
            #load_in_4bit = load_in_4bit,
            quantization_config = quantization_config,
            device_map = 'auto',
            trust_remote_code=True
        )
        model.config.use_cache = False
        model.config.pretrained_tp = 1
        
        self.model = model
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

    def load_adapter(self):
        if self.adapter_name == None:
            pass
        else:
            self.model = PeftModel.from_pretrained(
                self.model,self.adapter_name
            )

    
    def identify_tokens(self,class_tokens = None):

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


    def prompt_setup(self,prompt=None,prompt_path=None):
        if prompt == None:
            try:
                prompt = open(prompt_path).read()
            except:
                raise ValueError('Either prompt or prompt_path need to be specified')
        # TODO: Figure out if this is necessary

    

    def get_dataloader(
        self,
        input_text,
        batch_size=1,
    ):

        ds = GenerativeDataset(input_text,self.tokenizer)
        collator = DataCollatorWithPadding(self.tokenizer)
        dataloader = DataLoader(ds,batch_size=batch_size,collate_fn=collator)
        
        return dataloader

    def logit_extraction(self,input_data,tokens,batch_size=1,save=False):
        self.token_dict = self.identify_tokens(tokens)
        
        if type(input_data) == list:
            try:
                input_data = self.get_dataloader(input_data,batch_size=batch_size)
            except:
                raise ValueError('Batch size must be specified if text list is provided.')
        
        elif type(input_data) == str:
            input_data = self.get_dataloader([input_data],batch_size=batch_size)

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        preds = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(input_data):
    
                output = self.model(
                    input_ids = batch['input_ids'].to(device),
                    attention_mask = batch['attention_mask'].to(device)
                ).logits.detach().cpu()
    
                # Extract predictions following end of prompt
                generated = torch.vstack(
                    [output[i,batch['length'][i]-1,:] for i in range(len(batch['length']))]
                )
                preds.append(
                    softmax(generated[:,list(self.token_dict.values())],dim=-1).numpy()
                )
                if save != False:
                    pd.to_pickle(save)

        output_df = pd.DataFrame(np.vstack(preds),columns=list(self.token_dict.keys()))

        return output_df

    def text_generation(self,input_data,batch_size=1,max_new_tokens=100,save=False):
        self.tokenizer.padding_side = "left"
        if (type(input_data) == list) or (type(input_data) == pd.DataFrame):
            try:
                input_data = self.get_dataloader(input_data,batch_size=batch_size)
            except:
                raise ValueError('Batch size must be specified if text list is provided.')
        
        elif type(input_data) == str:
            input_data = self.get_dataloader([input_data],batch_size=batch_size)
        
        
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        preds = []
        self.model.eval()
        accelerator = Accelerator()
        self.model,input_data = accelerator.prepare(self.model,input_data)
        with torch.no_grad():
            for batch in tqdm(input_data):
                output = self.model.generate(
                    input_ids = batch['input_ids'].to('cuda'),
                    attention_mask = batch['attention_mask'].to('cuda'),
                    max_new_tokens = max_new_tokens
                )
    
                preds.extend(self.tokenizer.batch_decode(output))

                torch.cuda.empty_cache()

                if save != False:
                    pd.to_pickle(save)

        #self.tokenizer.padding_side = "right"
        #preds = [i.replace('</s>','').replace('<s>','') for i in preds]
        return preds

    def get_response_seq(self,response_seq):
        if type(response_seq) == list:
            return response_seq
        else:
            return self.tokenizer(response_seq)['input_ids'][1:]

    def trainer_setup(
        self,
        train_ds,
        response_seq,
        max_seq_length=8192,
        batch_size='auto',
        text_field = "prompt",
        lora_alpha = 16,
        lora_rank=32,
        lora_dropout=0.1,
        training_args = None,
        cache_dir = '~/.cache/TokenProbs'
    ):
        #TODO: set up custom batch size (without autofind)

        response_seq = self.get_response_seq(response_seq)

        peft_params = LoraConfig(
            lora_alpha = lora_alpha,
            r = lora_rank,
            lora_dropout = lora_dropout,
            bias = "none",
            task_type = "CAUSAL_LM"
        )
        
        if self.model.config.model_type == 'mistral':
            peft_params.target_modules = ["q_proj", "v_proj"]

        collator = DataCollatorForCompletionOnlyLM(
            response_seq, tokenizer = self.tokenizer,mlm=False
        )

        if training_args == None:
            training_args = TrainingArguments(
                output_dir = cache_dir,
                num_train_epochs = 1,
                auto_find_batch_size = True,
                gradient_accumulation_steps = 1,
                optim = "paged_adamw_32bit",
                logging_steps = 2,
                learning_rate = 2e-4,
                weight_decay = 0.001,
                max_grad_norm = 0.3,
                lr_scheduler_type="constant"
            )

        #max_seq_len = self.model.config.max_position_embeddings

        if type(train_ds) in [list,pd.DataFrame]:
            train_ds = GenerativeDataset(train_ds,self.tokenizer,train=True)
        
        self.trainer = SFTTrainer(
            model = self.model,
            train_dataset = train_ds,
            dataset_text_field=text_field,
            peft_config = peft_params,
            max_seq_length = max_seq_length,
            data_collator = collator,
            args = training_args,
            packing = False
        )

        accelerator = Accelerator()
        self.trainer = accelerator.prepare(self.trainer)
           

            
            
        

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