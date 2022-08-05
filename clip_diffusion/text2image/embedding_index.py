import faiss
import torch
from clip_diffusion.utils.functional import tensor_to_numpy


def build_embedding_index(
    embeddings_dir,
    index_path="embeddings.index",
    max_index_memory_usage="8GB",
    current_memory_available="16GB",
    metric_type="ip",
):
    """
    使用autofaiss產生最佳化的index
    embeddings_dir: 存放embedding的資料夾
    index_path: 輸出的index路徑
    max_index_memory_usage: 可用的記憶體上限
    current_memory_available: 目前可用的記憶體
    metric_type: query的similarity function(l2: L2 norm, ip: 內積)
    """

    import autofaiss

    autofaiss.build_index(
        embeddings=embeddings_dir,
        index_path=index_path,
        max_index_memory_usage=max_index_memory_usage,
        current_memory_available=current_memory_available,
        metric_type=metric_type,
    )


def load_faiss_index(index_path):
    """
    載入faiss index
    """

    return faiss.read_index(index_path)


def get_topk_results(index, embedding, topk=5):
    """
    從index中找出相似度前k高的距離(相似度)和indices
    """

    if isinstance(embedding, torch.Tensor):
        embedding = tensor_to_numpy(embedding)

    return index.search(embedding, topk)
