import torch

def predict_sequence_prob_esm1b_hf(seq, model):
    tokenizer = model.tokenizer_

    seq = " ".join(seq)

    token_ids = torch.tensor([tokenizer.encode(seq)])
    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
    output = model.model_(token_ids)
    output = torch.nn.LogSoftmax(dim=2)(output[0])
    sequence_output = output.cpu().detach().numpy()

    return sequence_output[0]

def get_esm1b_hf_embedding(seq, tokenizer, model):
    seq = " ".join(seq)

    token_ids = torch.tensor([tokenizer.encode(seq)])
    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
    output = model(token_ids)

    return output.logits.reshape((output.logits.shape[1], output.logits.shape[2])).cpu().detach().numpy()


def embed_seqs_esm1b_hf(
        seqs, model, tokenizer
):
    return {seq: [{'embedding': get_esm1b_hf_embedding(seq, tokenizer, model)}] for seq in seqs}


