from molbart.utils.tokenizers import ChemformerTokenizer
from molbart.models import BARTModel
from molbart.data.util import BatchEncoder
import torch
import numpy as np
import os
from tqdm import tqdm
from typing import Optional, List

VOCAB_PATH = os.path.join(os.path.dirname(__file__), "bart_vocab.json")

def checkpoint_path(model_name: str):
    return os.path.join(os.path.dirname(__file__), f"{model_name}.ckpt")

CHECKPOINT_DOWNLOAD_LINKS = {
    "Chemformer": "https://az.app.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq/folder/144881804954",
    "Chemformer_large": "https://az.app.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq/folder/144881806154",
}

class State:
    batch_encoder: Optional[BatchEncoder] = None
    model: Optional[BARTModel] = None

    model_name: Optional[str] = None
    device: Optional[str] = None
    batch_size: Optional[int] = None

    initialized: bool = False

def setup(model_name: str, device: str, batch_size: int) -> None:
    if device == "mps":
        raise ValueError(f"Device '{device}' is not supported.")

    tokenizer = ChemformerTokenizer(filename=VOCAB_PATH)
    batch_encoder = BatchEncoder(tokenizer=tokenizer, masker=None, max_seq_len=-1)
    if not os.path.isfile(checkpoint_path(model_name)):
        raise RuntimeError(f"Download checkpoint '{CHECKPOINT_DOWNLOAD_LINKS[model_name]}' and save it as '{checkpoint_path(model_name)}'.")
    
    model = BARTModel.load_from_checkpoint(checkpoint_path(model_name), decode_sampler=None, vocabulary_size=len(tokenizer.vocabulary))
    model = model.to(device)
    model.eval()
    State.batch_encoder = batch_encoder
    State.model = model
    State.model_name = model_name
    State.device = device
    State.batch_size = batch_size
    State.initialized = True

def encode(smiles: List[str]) -> np.ndarray:
    if not State.initialized:
        raise RuntimeError("Service is not setup, call 'setup' before 'encode'.")

    outputs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(smiles), State.batch_size), f"Encoding with {State.model_name}"):
            smiles_batch = smiles[i:i+State.batch_size]
            x, mask = State.batch_encoder(smiles_batch)
            x = x.to(State.device)
            mask = mask.to(State.device)
            batch = {"encoder_input": x, "encoder_pad_mask": mask}
            out = State.model.encode(batch)
            mul_mask = (~mask).unsqueeze(-1)
            counts = mul_mask.sum(dim=0)
            embeddings = (out * mul_mask).sum(dim=0) / counts
            outputs.append(embeddings)
    outputs = torch.concat(outputs)
    outputs = outputs.cpu().numpy()
    return outputs
