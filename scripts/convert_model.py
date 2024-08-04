from contextlib import ExitStack
from dataclasses import dataclass
from io import FileIO
import json
import os
from torch import Tensor
from safetensors.torch import load_file
import logging
import struct
from tqdm import trange
import gc

import torch
logger = logging.getLogger(__name__)


logging.basicConfig(level=logging.DEBUG)


DEFAULT_MODEL_PATH = "./model"
DEFAULT_OUTPUT_PATH = "./model/model.bin"
FORCE_COVER = True



def _search_model_safetensors_files(dirpath=DEFAULT_MODEL_PATH) -> list[str]:
    safetensor_files:list[str] = []
    for filepath in os.listdir(dirpath):
        if filepath.startswith('model') and filepath.endswith('safetensors'):
            safetensor_files.append(os.path.join(dirpath, filepath))
    return safetensor_files

def _get_layer_index(item:tuple[str,Tensor]) -> float:
    if "layer" in item[0]:
        index = float(item[0].split('.')[2]) / 100
        # pre/post_attention_norm: RMSnorm
        if   "input_layernorm" in item[0]:
            index += 0
        elif "self_attn.q_proj" in item[0]:
            index += 1
        elif "self_attn.k_proj" in item[0]:
            index += 2
        elif "self_attn.v_proj" in item[0]:
            index += 3
        elif "self_attn.o_proj" in item[0]:
            index += 4
        elif "post_attention_layernorm" in item[0]:
            index += 5
        elif "pre_feedforward_layernorm" in item[0]:
            index += 6
        elif "mlp.down_proj" in item[0]:
            index += 7
        elif "mlp.gate_proj" in item[0]:
            index += 8
        elif "mlp.up_proj" in item[0]:
            index += 9
        elif "post_feedforward_layernorm" in item[0]:
            index += 10
        else:
            logger.error(f"Can not recognize layer name: {item[0]}")
            raise RuntimeError
        return index

    else:
        if "embed_tokens" in item[0]:
            return -1.0
        else:
            assert "norm" in item[0]
            return 11.0


def sort_model(model:list[tuple[str,Tensor]]):
    model.sort(key=_get_layer_index)


def write_to_file(output_fp:FileIO, model:list[tuple[str,Tensor]], max_chunk_size=1024*1024):
    for name, tensor in model:
        tensor = tensor.to(torch.float32).view(-1)
        size = tensor.numel()
        logger.info(f"Writing {name}")
        for i in trange(0, size, max_chunk_size):
            chunk = tensor[i:i+max_chunk_size].tolist()
            packed_chunk = struct.pack(f"{len(chunk)}f", *chunk)
            output_fp.write(packed_chunk)
            del chunk, packed_chunk
            gc.collect()

    logger.info("Finish writing.")



def main():
    logger.info(msg="Start converting model.")

    # check path
    if os.path.exists(DEFAULT_OUTPUT_PATH):
        if FORCE_COVER:
            os.remove(DEFAULT_OUTPUT_PATH)
            logger.warn(f"{DEFAULT_OUTPUT_PATH} will be overwritten.")
        else:
            logger.error(f"A file with same name {DEFAULT_OUTPUT_PATH} in the path.")
            raise RuntimeError()

    output_fp = open(DEFAULT_OUTPUT_PATH, "wb", buffering=0)

    head_str = "GRMD google/gemma-2-2b-it"
    output_fp.write(struct.pack(f"I{len(head_str)}s", len(head_str), head_str.encode('utf-8')))

    model:list[tuple[str,Tensor]] = []   
    for path in _search_model_safetensors_files():
        part = load_file(path)
        for name,tensor in part.items():
            model.append((name, tensor))


    sort_model(model)

    write_to_file(output_fp, model)

    output_fp.close()

    logger.info("Finish converting model.")
    


if __name__ == "__main__":
    main()