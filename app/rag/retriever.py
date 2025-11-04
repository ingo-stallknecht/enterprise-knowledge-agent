# app/rag/retriever.py
from .embedder import Embedder
from .index import DocIndex
import numpy as np


class Retriever:
def __init__(self, model_name: str, normalize: bool, index_path: str, store_path: str):
self.embedder = Embedder(model_name, normalize)
self.docindex = DocIndex(index_path, store_path).load()
def retrieve(self, query: str, k: int = 6):
vec = self.embedder.encode([query])
return self.docindex.search(vec, k)