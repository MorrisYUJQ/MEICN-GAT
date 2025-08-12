import numpy as np

def recall_at_k(r, k, all_pos_num):
    """
    Calculate recall@k.
    计算召回率@k。

    Args:
        r (list): A list of 0/1 indicating hits. / 0/1命中指示列表。
        k (int): The "k" in "recall@k". / "recall@k"中的"k"。
        all_pos_num (int): The total number of positive items for the user. / 用户的正向物品总数。

    Returns:
        float: The recall@k value.
    """
    r = np.asfarray(r)[:k]
    if all_pos_num == 0:
        return 0.0
    return np.sum(r) / all_pos_num

def precision_at_k(r, k):
    """
    Calculate precision@k.
    计算精确率@k。

    Args:
        r (list): A list of 0/1 indicating hits. / 0/1命中指示列表。
        k (int): The "k" in "precision@k".

    Returns:
        float: The precision@k value.
    """
    r = np.asfarray(r)[:k]
    return np.mean(r)

def hit_at_k(r, k):
    """
    Calculate hit_ratio@k.
    计算命中率@k。

    Args:
        r (list): A list of 0/1 indicating hits. / 0/1命中指示列表。
        k (int): The "k" in "hit_ratio@k".

    Returns:
        float: The hit_ratio@k value.
    """
    r = np.asfarray(r)[:k]
    if np.sum(r) > 0:
        return 1.0
    else:
        return 0.0

def dcg_at_k(r, k):
    """
    Calculate discounted cumulative gain (DCG).
    计算折扣累积增益 (DCG)。

    Args:
        r (list): A list of 0/1 indicating hits. / 0/1命中指示列表。
        k (int): The "k" in "dcg@k".

    Returns:
        float: The DCG value.
    """
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(r, k):
    """
    Calculate normalized discounted cumulative gain (NDCG@k).
    计算归一化折扣累积增益 (NDCG@k)。

    Args:
        r (list): A list of 0/1 indicating hits. / 0/1命中指示列表。
        k (int): The "k" in "ndcg@k".

    Returns:
        float: The NDCG@k value.
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max
