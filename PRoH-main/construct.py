import networkx as nx

import os
import json
import time
from PRoH import PRoH
from PRoH.utils import logger
import argparse
import asyncio
from tqdm.asyncio import tqdm_asyncio
import numpy as np
import shutil

API_KEY = open("openai_api_key.txt").read().strip()
BASE_URL = open("openai_base_url.txt").read().strip()

os.environ["OPENAI_API_KEY"] = API_KEY

embedding_with_base = EmbeddingFunc(
    embedding_dim=openai_embedding.embedding_dim,
    max_token_size=openai_embedding.max_token_size,
    func=partial(
        openai_embedding.func,
        base_url=BASE_URL,
        api_key=API_KEY,
    ),
)

parser = argparse.ArgumentParser()
parser.add_argument('--data_source', default='example')
parser.add_argument("--mdl", type=str, default="gpt-4o-mini")
parser.add_argument("--threshold", type=float, default=0.9)
parser.add_argument("--k", type=int, default=10)

args = parser.parse_args()
data_source = args.data_source
model = args.mdl
threshold = args.threshold
top_k = args.k

similar_params={
    "similar_threshold": threshold,  
    "similar_top_k": top_k      
}


CONTEXT_FILE = os.path.abspath(os.path.join("contexts", f"{data_source}_contexts.json"))



WORKING_DIR = os.path.abspath(f"./data/work/{data_source}")
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

LOG_FILE = os.path.join(WORKING_DIR, f"{data_source}_construct.log")



rag = PRoH(
    working_dir=WORKING_DIR,
    embedding_func_max_async=32,
    llm_model_max_async=8,
    llm_model_name=model,
    log_level="INFO",
    log_file=LOG_FILE,
    max_concurrency=4,
    similar_params=similar_params,
    llm_model_kwargs={
        "base_url": BASE_URL,
        "api_key": API_KEY,
    },
    embedding_func=embedding_with_base,
)


def initial_insert_text(rag, file_path):
    with open(file_path, mode="r") as f:
        unique_contexts = json.load(f)

    retries = 0
    max_retries = 10
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")


if __name__ == "__main__":

    initial_insert_text(rag, CONTEXT_FILE)
    # make a copy of the .graphml file under WORKING_DIR
    graph_path = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
    backup_path = os.path.join(WORKING_DIR, "graph_chunk_entity_relation_init.graphml")
    shutil.copy2(graph_path, backup_path)
    logger.info(f"Backup created: {backup_path}")


    rag.init_entity_names_vdb()
    rag.add_similar_edges()

    backup_path = os.path.join(WORKING_DIR, f"graph_chunk_entity_relation_{similar_params['similar_threshold']}_{similar_params['similar_top_k']}.graphml")
    shutil.copy2(graph_path, backup_path)
    logger.info(f"Backup created: {backup_path}")

    rag.merge_synonyms()










