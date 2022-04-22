import torch
from transformers import ESMForMaskedLM, ESMTokenizer

class ESM_HF(object):
    def __init__(self, model_path):
        self.name_ = 'esm1b_huggingface'

        model = ESMForMaskedLM.from_pretrained(model_path, ignore_mismatched_sizes=True)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self.model_ = model

        self.tokenizer_ = ESMTokenizer.from_pretrained("facebook/esm-1b", do_lower_case=False)
        self.alphabet_ = {e:i for i,e in enumerate(self.tokenizer_.all_tokens)}
        #self.alphabet_['J'] = self.alphabet_['<unk>']
        self.unk_idx_ = self.alphabet_['<unk>']

        self.vocabulary_ = {
            tok: self.alphabet_[tok]
            for tok in self.alphabet_ if '<' not in tok
        }
