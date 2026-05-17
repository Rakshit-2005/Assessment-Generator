"""Simple Hugging Face Inference API embeddings adapter.

This calls the HF Inference API (feature-extraction) for models like
`sentence-transformers/all-MiniLM-L6-v2`. Set environment variable
`HF_API_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` with your token.

The adapter returns a numpy array of shape (n_texts, dim).
"""
from typing import List
import os
import json
import requests
import numpy as np

HF_TOKEN = os.getenv('HF_API_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
# Default model to call for embeddings; can be changed when constructing.
DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


class HfEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL, token: str = None):
        self.model = model
        self.token = token or HF_TOKEN
        if not self.token:
            raise RuntimeError('HF_API_TOKEN not set')
        self.endpoint = f'https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model}'
        self.headers = {'Authorization': f'Bearer {self.token}', 'Content-Type': 'application/json'}

    def embed(self, texts: List[str]):
        # API supports a single input or a batch; pass list for batch
        payload = json.dumps({
            'inputs': texts,
            # use default params; keep it simple
        })
        resp = requests.post(self.endpoint, headers=self.headers, data=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f'HF inference failed: {resp.status_code} {resp.text}')
        data = resp.json()
        # data may be a list of lists (for batch) or nested; coerce to numpy
        arr = np.array(data, dtype=float)
        # If the API returned embeddings for each token (2D per input), try to average
        if arr.ndim == 3:
            # shape (n_inputs, seq_len, dim) -> average over seq_len
            arr = arr.mean(axis=1)
        return arr
