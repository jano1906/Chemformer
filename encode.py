from argparse import ArgumentParser
from molbart.utils.tokenizers import ChemformerTokenizer
from molbart.models import BARTModel
from molbart.data.util import BatchEncoder
import math
import torch
import numpy as np
import os

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    return parser

def main(args):
    VOCAB_PATH = os.path.join(os.path.dirname(__file__), "bart_vocab.json")
    BATCH_SIZE = 32

    with open(args.input) as f:
        data = f.readlines()
    tokenizer = ChemformerTokenizer(filename=VOCAB_PATH)
    tokenized_data = tokenizer(data)
    max_len = max(len(x) for x in tokenized_data)
    batch_encoder = BatchEncoder(tokenizer=tokenizer, masker=None, max_seq_len=2 ** math.ceil(math.log2(max_len)))
    model = BARTModel.load_from_checkpoint(args.model_path, decode_sampler=None, vocabulary_size=len(tokenizer.vocabulary))
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(data), BATCH_SIZE):
            data_batch = data[i:i+BATCH_SIZE]
            x, mask = batch_encoder(data_batch)
            batch = {"encoder_input": x, "encoder_pad_mask": mask}
            out = model.encode(batch)
            mul_mask = (~mask).unsqueeze(-1)
            counts = mul_mask.sum(dim=0)
            embeddings = (out * mul_mask).sum(dim=0) / counts
            outputs.append(embeddings)
    outputs = torch.concat(outputs)
    outputs = outputs.cpu().numpy()
    with open(args.output, "wb") as f:
        np.save(f, outputs)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    