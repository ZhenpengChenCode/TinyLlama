import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Pool, cpu_count

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Tokenizer

import pandas as pd


def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    split: str="train",
    filenames_subset: List[str] = None,
    process_id: int = 0
) -> None:
    import zstandard as zstd

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset 
    
    if not filenames:
        raise RuntimeError(
            f"No files matching  found at {source_path}. \n"
            "Make sure you download the data..."
        )

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_starcoder_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    for filepath in filenames:
        print(f"Processing {filepath}")
        try:
            contents = pd.read_parquet(filepath, engine='pyarrow')['content']
        except:
            print(f"Error reading {filepath}!!")
            continue
        for text in contents:
            text_ids = tokenizer.encode(text)
            builder.add_array(np.array(text_ids, dtype=builder.dtype))

    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids, see https://github.com/jzhang38/TinyLlama/issues/83 for more details
    # builder.write_reminder()


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    destination_path: Path = Path("data/red_pajama_sample"),
    chunk_size: int = 2049 * 1024,
    split: str="train",
    percentage: float = 1.0,
    filenames_subset: List[str] = None,
) -> None:
    import time
    assert split == "train" #  starcoder only has train data
    filenames = glob.glob(os.path.join(source_path, "*/*.parquet"), recursive=True)
    # only retrain subsets that follow the prefix in filenames_subset
    if filenames_subset:
        filenames = [f for f in filenames if any([prefix in f for prefix in filenames_subset])]
    filenames = filenames[:int(len(filenames) * percentage)]
    max_files_per_process = 20
    num_process = 4
    total_processes = int(len(filenames) / max_files_per_process)
    chunked_filenames = np.array_split(filenames, total_processes)
    
    p = Pool(num_process)
    
    # processes = []
    start_time = time.time()
    for i, subset in enumerate(chunked_filenames):
        p.apply_async(prepare_full, args=(source_path, tokenizer_path, destination_path, chunk_size, split, list(subset), i))
    print("Waiting for all subprocesses done...")
    p.close()
    p.join()
    print('All subprocesses done.')

    # for i, subset in enumerate(chunked_filenames):
    #     p = Process(target=prepare_full, args=(source_path, tokenizer_path, destination_path, chunk_size, split, list(subset), i))
    #     processes.append(p)
    #     p.start()

    # for p in processes:
    #     p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    #from jsonargparse import CLI
    #CLI(prepare)
    #python3 scripts/prepare_starcoder.py --source_path /path/to/starcoderdata/ --tokenizer_path data/llama --destination_path data/slim_star_combined --split train --percentage 1.0
    #python3 scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama --tokenizer_path data/llama  --destination_path data/slim_star_combined --split validation --percentage 1.0
    #python3 scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama --tokenizer_path data/llama  --destination_path data/slim_star_combined --split train --percentage 1.0
    prepare(source_path=Path("/home/baidu/dataset/starcoderdata"),
            tokenizer_path=Path("/home/baidu/workspace/TinyLlama-1.1B-intermediate-step-480k-1T"),
            destination_path=Path("/home/baidu/workspace/TinyLlama/data/slim_star_combined"),
            split='train',
            percentage=1.0)
