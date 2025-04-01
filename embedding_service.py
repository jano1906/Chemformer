from molbart.utils.tokenizers import ChemformerTokenizer
from molbart.models import BARTModel
from molbart.data.util import BatchEncoder
import torch
import numpy as np
import os
from tqdm import tqdm
from typing import Optional, List

VOCAB_PATH = os.path.join(os.path.dirname(__file__), "bart_vocab.json")
MODEL_PATHS = {
    "chemformer": os.path.join(os.path.dirname(__file__), "chemformer.ckpt"),
    "chemformer_large": os.path.join(os.path.dirname(__file__), "chemformer_large.ckpt"),
}

class State:
    batch_encoder: Optional[BatchEncoder] = None
    model: Optional[BARTModel] = None
    batch_size: Optional[int] = None

def setup(model: str, device: str, batch_size: int) -> None:
    tokenizer = ChemformerTokenizer(filename=VOCAB_PATH)
    batch_encoder = BatchEncoder(tokenizer=tokenizer, masker=None, max_seq_len=-1)
    model = BARTModel.load_from_checkpoint(MODEL_PATHS[model], decode_sampler=None, vocabulary_size=len(tokenizer.vocabulary))
    model = model.to(device)
    model.eval()
    State.batch_encoder = batch_encoder
    State.model = model
    State.batch_size = batch_size

def encode(smiles: List[str]) -> np.ndarray:
    if any([State.batch_encoder is None, State.model is None, State.batch_size is None]):
        raise RuntimeError("Service is not setup, call 'setup' before 'encode'.")

    outputs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(smiles), State.batch_size), "Encoding..."):
            smiles_batch = smiles[i:i+State.batch_size]
            x, mask = State.batch_encoder(smiles_batch)
            batch = {"encoder_input": x, "encoder_pad_mask": mask}
            out = State.model.encode(batch)
            mul_mask = (~mask).unsqueeze(-1)
            counts = mul_mask.sum(dim=0)
            embeddings = (out * mul_mask).sum(dim=0) / counts
            outputs.append(embeddings)
    outputs = torch.concat(outputs)
    outputs = outputs.cpu().numpy()
    return outputs
