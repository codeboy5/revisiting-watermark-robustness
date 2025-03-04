import math
import json
import torch
import jsonlines
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import sent_tokenize

from torch import FloatTensor, LongTensor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate
from rich import print as rprint
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from transformers import T5Tokenizer, T5ForConditionalGeneration

from kgw_watermarking.watermark_reliability_release.watermark_processor import WatermarkDetector, WatermarkLogitsProcessor, LogitsProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

# just to check the generation length
original_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b", device_map="auto")
if original_tokenizer.pad_token is None:
    original_tokenizer.pad_token = original_tokenizer.eos_token

attack_tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl", device_map="auto")
if attack_tokenizer.pad_token is None:
    attack_tokenizer.pad_token = attack_tokenizer.eos_token

tokenizer = attack_tokenizer

vocab = tokenizer.get_vocab().values()
vocab_size = len(vocab)

print("Len Vocab: ", vocab_size)
print("Device: ", device)

#! Watermark strength hyperparameters
gamma = 0.25
delta = 2.0

unigram_freq_openwebtext = torch.load("./computed_data/unigram_freq_openwebtext_t5.pt")
relative_freq_openwebtext = unigram_freq_openwebtext / unigram_freq_openwebtext.sum() # this is the distribution of the human text

predicted_split = torch.load("./computed_data/Pythia-1.4b_kgw_simple0_predicted_split.pt")

dipper_tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
dipper = T5ForConditionalGeneration.from_pretrained("kalpeshk2011/dipper-paraphraser-xxl",
                                                    device_map="auto", 
                                                    torch_dtype=torch.float16)

dipper.eval()
print("Loaded the paraphrasing model!")


class CustomLogitsProcessor(LogitsProcessor):
    """
    Currently just implemented this for the fixed green and red list watermark. We give our estimated green and red list and to evade the detection we will boost the logits of the 
    red tokens. So the idea would be to boost the red tokens so that the text has more red tokens and we can decrease the watermark signal in the text and also keep the semantics 
    intact.
    """
    def __init__(
        self,
        vocab,
        delta,
        greenlist_ids,
        redlist_ids,
        ):
        super().__init__()

        # watermarking parameters
        self.vocab = vocab
        self.vocab_size = len(vocab)
        # self.gamma = gamma
        self.delta = delta

        # we have estimated the green and red list using the frequency analysis
        self.greenlist_ids = greenlist_ids
        self.redlist_ids = redlist_ids

    def _calc_greenlist_mask(self, scores: torch.FloatTensor) -> torch.BoolTensor:
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(green_tokens_mask)):
            green_tokens_mask[b_idx][self.greenlist_ids] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask
    
    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] - greenlist_bias
        return scores

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        green_tokens_mask = self._calc_greenlist_mask(scores=scores)
        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)
        return scores


def paraphrase(input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
    """Paraphrase a text using the DIPPER model.

    Args:
        input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
        lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
        order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
        **kwargs: Additional keyword arguments like top_p, top_k, max_length.
    """
    assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
    assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

    lex_code = int(100 - lex_diversity)
    order_code = int(100 - order_diversity)

    input_text = " ".join(input_text.split())
    sentences = sent_tokenize(input_text)
    prefix = " ".join(prefix.replace("\n", " ").split())
    output_text = ""

    for sent_idx in range(0, len(sentences), sent_interval):
        curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
        final_input_text = f"lexical = {lex_code}, order = {order_code}"
        if prefix:
            final_input_text += f" {prefix}"
        final_input_text += f" <sent> {curr_sent_window} </sent>"

        final_input = dipper_tokenizer([final_input_text], return_tensors="pt")
        final_input = {k: v.to(device) for k, v in final_input.items()}

        with torch.inference_mode():
            outputs = dipper.generate(**final_input, **kwargs)
        outputs = dipper_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        prefix += " " + outputs[0]
        output_text += " " + outputs[0]

    return output_text

input_file_test = "./generated_text/Pythia-1.4b_kgw_simple0_c4_valset.json"
with open(input_file_test, "r") as f:
    data_test = json.load(f)

data_test = data_test['samples'][f'kgw_gamma{gamma}_delta{delta}_schemesimple_0_pythia-1.4b']
# data_test = data_test['samples'][f'kgw_gamma{gamma}_delta{delta}_schemesimple_0_Llama-2-7b-hf']

STEP_KEYS = ['step_1', 'step_20', 'step_1000']
LEX_VALUES = [20, 40, 60]
NUM_PARAPHRASES = 200

#! should we also just do the random attack here to check those number's look?
paraphrased_results = []
for i in tqdm(range(len(data_test['prompt_text']))):

    # remove completions that are too small
    tokens = original_tokenizer(data_test['model_text'][i], return_tensors="pt")['input_ids']
    if tokens.shape[1] <= 200:
        continue
    
    current_results = {}
    current_results["prompt_text"] = data_test['prompt_text'][i]
    current_results['full_human_text'] = data_test['full_human_text'][i]
    current_results['full_model_text'] = data_test['full_model_text'][i]

    for lex_value in LEX_VALUES:

        # let's paraphrase without any knowledge of the green and red list
        current_results[f'paraphrased_raw_L{lex_value}'] = paraphrase(data_test['model_text'][i], lex_diversity=lex_value, order_diversity=0, 
                                                prefix=data_test['prompt_text'][i], do_sample=True, top_p=0.9, top_k=None, 
                                                max_length=512, sent_interval=2)
        
        for key in STEP_KEYS:
            greenlist_ids = list(predicted_split[key]['greenlist_ids'])
            redlist_ids = list(predicted_split[key]['redlist_ids'])

            #! should be somwhow mid boost that shit?
            # greenlist_ids = list(filter(lambda x : x != 1, greenlist_ids))
            # redlist_ids = list(filter(lambda x : x != 1, redlist_ids))

            reverse_watermark = CustomLogitsProcessor(
                                    vocab=dipper_tokenizer.get_vocab().values(), 
                                    delta=1.0, greenlist_ids=greenlist_ids, redlist_ids=redlist_ids)

            current_results[f'paraphrased_using_{key}_L{lex_value}'] = paraphrase(data_test['model_text'][i], lex_diversity=lex_value, order_diversity=0, 
                            prefix=data_test['prompt_text'][i], do_sample=True, top_p=0.9, 
                            top_k=None, max_length=512, logits_processor=LogitsProcessorList([reverse_watermark]), sent_interval=2)  
        
    paraphrased_results.append(current_results)

    if len(paraphrased_results) >= NUM_PARAPHRASES:
        break

file_path_save = "./computed_data/Pythia-1.4b-hf_reverse_paraphrase_c4_valset_dipper_attack_delta1.json"
with open(file_path_save, "w") as file:
    json.dump(paraphrased_results, file, indent=4)