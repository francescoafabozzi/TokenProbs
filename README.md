# TokenProbs

Extract token-level probability scores from generative language models (GLMs) without fine-tuning. Often times, it is relevent to request probability assessment to binary or multi-class outcomes. GLMs are not well-suited for this task. Instead, use `LogitExtractor` to obtain label probabilities without fine-tuning.


## Installation

```bash
conda create -n GenCasting python=3.9
pip install GenCasting 
```

## Usage
```python
from TokenProbs import LogitExtractor

extractor = LogitExtractor(
    model_name = 'mistralai/Mistral-7B-Instruct-v0.1',
    quantization="8bit" # None = Full precision, "4bit" also suported
)

# Prompt sentence with 
prompt = "Instructions: What is the sentiment of this news article? Select from {positive/neutral/negative}.\nInput: %text_input\nAnswer: "
sentence = "AAPL shares were up in morning trading, but closed even on the day."
prompted_sentence = prompt.replace("%text_input",sentence)

pred_tokens = ['positive','neutral','negative']

probabilities = extractor.logit_extraction(prompted_sentence,pred_tokens)

# Display the probabilities
print(f"Probabilities: {probabilities}")
Probabilities: {'positive': 0.7, 'neutral': 0.2, 'negative': 0.1}
```

## Usage



