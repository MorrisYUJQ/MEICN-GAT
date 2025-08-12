import torch
import numpy as np
import multiprocessing
from tqdm import tqdm
import faiss
from .metrics import recall_at_k, ndcg_at_k, hit_at_k, precision_at_k

def test_multi(user_emb, item_emb, user_dict, n_params, Ks, device):
    """
    Test the model's performance on the test set.
    在测试集上测试模型的性能。

    Args:
        user_emb (torch.Tensor): User embeddings. / 用户嵌入。
        item_emb (torch.Tensor): Item embeddings. / 物品嵌入。
        user_dict (dict): Dictionary containing train and test user-item interactions. / 包含训练和测试用户-物品交互的字典。
        n_params (dict): Dictionary of model parameters (e.g., n_users, n_items). / 模型参数字典。
        Ks (list): List of K values for top-K evaluation. / 用于top-K评估的K值列表。
        device (torch.device): The device to run the evaluation on. / 运行评估的设备。

    Returns:
        dict: A dictionary containing the evaluation results (recall, ndcg, etc.). / 包含评估结果的字典。
    """
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']
    n_test_users = len(test_user_set)
    
    # Use Faiss for efficient similarity search
    # 使用Faiss进行高效的相似度搜索
    item_faiss = item_emb.cpu().numpy().astype('float32')
    index = faiss.IndexFlatL2(item_faiss.shape[1])
    index.add(item_faiss)
    
    result = {
        "precision": np.zeros(len(Ks)),
        "recall": np.zeros(len(Ks)),
        "ndcg": np.zeros(len(Ks)),
        "hit_ratio": np.zeros(len(Ks)),
    }

    pool = multiprocessing.Pool(cores)

    u_batch_size = 2048
    i_batch_size = 2048

    test_users = list(test_user_set.keys())
    n_user_batchs = len(test_users) // u_batch_size + 1
    
    count = 0
    
    with torch.no_grad():
        for u_batch_id in tqdm(range(n_user_batchs), desc="Testing"):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = test_users[start:end]
            
            # Get ground truth items for this batch
            # 获取该批次的真实物品
            user_pos_test = [test_user_set[u] for u in user_batch]
            
            # Perform search to get top-K recommendations
            # 执行搜索以获得top-K推荐
            _, topk_items_batch = index.search(user_emb[user_batch].cpu().numpy(), max(Ks))
            
            # Calculate metrics for this batch
            # 计算该批次的指标
            batch_result = []
            for i in range(len(user_batch)):
                r = []
                for item in topk_items_batch[i]:
                    if item in user_pos_test[i]:
                        r.append(1)
                    else:
                        r.append(0)
                
                re = get_performance(user_pos_test[i], r, Ks)
                batch_result.append(re)

            for re in batch_result:
                result["precision"] += re["precision"]
                result["recall"] += re["recall"]
                result["ndcg"] += re["ndcg"]
                result["hit_ratio"] += re["hit_ratio"]
            
            count += len(user_batch)

    assert count == n_test_users
    pool.close()
    
    result["precision"] /= n_test_users
    result["recall"] /= n_test_users
    result["ndcg"] /= n_test_users
    result["hit_ratio"] /= n_test_users
    
    return result


def get_performance(user_pos_test, r, Ks):
    """
    Calculate performance metrics for a single user.
    计算单个用户的性能指标。

    Args:
        user_pos_test (list): List of ground truth items for the user. / 用户的真实物品列表。
        r (list): List of 0/1 indicating whether a recommended item is a hit. / 0/1列表，指示推荐物品是否命中。
        Ks (list): List of K values for top-K evaluation. / 用于top-K评估的K值列表。

    Returns:
        dict: A dictionary containing the performance metrics for this user. / 包含该用户性能指标的字典。
    """
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))

    return {
        "recall": np.array(recall),
        "precision": np.array(precision),
        "ndcg": np.array(ndcg),
        "hit_ratio": np.array(hit_ratio),
    }
