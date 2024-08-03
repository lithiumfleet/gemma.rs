from contextlib import ExitStack
from io import FileIO
import json
import os
from torch import Tensor
from safetensors.torch import load_file
import logging
import struct
import gc
logger = logging.getLogger(__name__)


logging.basicConfig(level=logging.DEBUG)


DEFAULT_MODEL_PATH = "./model"
DEFAULT_OUTPUT_PATH = "./model/model.bin"



def _get_model_safetensors_files(dirpath=DEFAULT_MODEL_PATH) -> list[str]:
    safetensor_files:list[str] = []
    for filepath in os.listdir(dirpath):
        if filepath.startswith('model') and filepath.endswith('safetensors'):
            safetensor_files.append(os.path.join(dirpath, filepath))
    return safetensor_files

def write_to_file(output_fp:FileIO, model:dict[str,Tensor]):
    raise NotImplementedError



def main():
    logger.info(msg="Start converting model.")

    # check path
    if os.path.exists(DEFAULT_OUTPUT_PATH):
        logger.error(f"A file with same name {DEFAULT_OUTPUT_PATH} in the path.")
        raise RuntimeError()

    output_fp = open(DEFAULT_OUTPUT_PATH, "wb", buffering=0)

    model_parts = []
    for path in _get_model_safetensors_files():
        model_parts.append(load_file(path))

    model:dict[str,Tensor] = model_parts[0]    
    for i in range(1,len(model_parts)):
        model.update(model_parts[i])

    write_to_file(output_fp, model)

    output_fp.close()

    logger.info("Finish converting model.")
    


if __name__ == "__main__":
    main()