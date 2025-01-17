import torch
from transformers import BertForMaskedLM, BertTokenizer

class ProtBertModel(object):
    def __init__(self, model_path):
        self.name_ = 'protbert'

        model = BertForMaskedLM.from_pretrained(model_path, ignore_mismatched_sizes=True)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self.model_ = model

        self.tokenizer_ = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.alphabet_ = self.tokenizer_.vocab
        #self.alphabet_['J'] = self.alphabet_['<unk>']
        self.unk_idx_ = self.alphabet_['[UNK]']

        self.vocabulary_ = {
            tok: self.alphabet_[tok]
            for tok in self.alphabet_ if '[' not in tok
        }
