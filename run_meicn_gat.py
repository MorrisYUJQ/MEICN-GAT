__author__ = "Jiaqian Yu"

import argparse
import gc
import os
import pickle
import random
from collections import defaultdict
from time import time

import dgl
import numpy as np
import torch
from dgl.dataloading import MultiLayerNeighborSampler
from prettytable import PrettyTable
# Note: The following imports assume that the 'src' and 'data' directories are in the python path.
# This can be achieved by running the script from the root of the project.
# 注意：以下导入假定'src'和'data'目录位于Python路径中。
# 这可以通过从项目根目录运行脚本来实现。
from src.base_model import M2GNN_word
from src.evaluate import test_multi
from src.parser import parse_args


def load_data(path, dataset_name):
    """
    Load graph, training, and testing data from preprocessed pkl files.
    从预处理的pkl文件中加载图、训练和测试数据。

    Args:
        path (str): The directory where the data files are stored. / 数据文件所在的目录。
        dataset_name (str): The name of the dataset. / 数据集的名称。

    Returns:
        tuple: A tuple containing training data, testing data, user interaction dictionary, 
               model parameters, and the graph.
               一个元组，包含训练数据、测试数据、用户交互字典、模型参数和图。
    """
    graph_path = os.path.join(path, dataset_name, "graph_tag.pkl")
    with open(graph_path, "rb") as f:
        graph_dp = pickle.load(f)
        graph_dp_tag = pickle.load(f)
    print("Graph loaded successfully.")
    del graph_dp
    gc.collect()

    train_test_path = os.path.join(path, dataset_name, "train_test.pkl")
    with open(train_test_path, "rb") as f:
        train_cf = pickle.load(f)
        test_cf = pickle.load(f)
        len_item = pickle.load(f)
        train_user_set = pickle.load(f)
        test_user_set = pickle.load(f)
    print("Training and testing sets loaded successfully.")

    # Extract graph statistics and model parameters
    # 提取图的统计信息和模型参数
    n_users = graph_dp_tag.num_nodes(ntype="h")
    n_items = graph_dp_tag.num_nodes(ntype="u")
    n_item4rs = len_item
    n_tag = graph_dp_tag.num_nodes(ntype="t")

    n_params = {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_items4rs": int(n_item4rs),
        "n_tag": int(n_tag),
    }
    user_dict = {"train_user_set": train_user_set, "test_user_set": test_user_set}

    return train_cf, test_cf, user_dict, n_params, graph_dp_tag


if __name__ == "__main__":
    # --- Fix Random Seed ---
    # --- 固定随机种子 ---
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Parse Arguments ---
    # --- 解析命令行参数 ---
    args = parse_args()
    
    # --- Load Data ---
    # --- 加载数据 ---
    # The data path is now specified via command line argument.
    # 数据路径现在通过命令行参数指定。
    train_cf, test_cf, user_dict, n_params, graph_tag = load_data(args.data_path, args.dataset)
    
    # Adjust batch size based on graph properties to normalize training steps.
    # 根据图的属性调整批量大小，以标准化训练步骤。
    args.batch_size = int(
        args.batch_size
        / graph_tag.num_edges("interaction")
        * (graph_tag.num_edges("t2t") + graph_tag.num_edges("interaction"))
    )

    # --- Build Dataloaders ---
    # --- 构建数据加载器 ---
    # Training dataloader
    # 训练数据加载器
    train_eids = {
        "interaction": torch.tensor(range(graph_tag.num_edges(etype="interaction"))),
        "t2t": torch.tensor(range(graph_tag.num_edges(etype="t2t"))),
    }
    num_nei = args.max_len
    sampler_dict = {
        "h_k_t": num_nei,
        "h_u_k_t": num_nei,
        "t2t": num_nei,
        "u_k_t": num_nei,
        "interaction": 0,
        "test": 0,
    }
    sampler = MultiLayerNeighborSampler([sampler_dict] * 2)
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)
    dataloader = dgl.dataloading.EdgeDataLoader(
        graph_tag,
        train_eids,
        sampler,
        device=device,
        negative_sampler=neg_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        use_ddp=False,
        drop_last=False,
    )

    # Testing dataloader
    # 测试数据加载器
    test_eids = {"test": torch.tensor(range(graph_tag.num_edges(etype="test")))}
    test_dataloader = dgl.dataloading.EdgeDataLoader(
        graph_tag,
        test_eids,
        sampler,
        device=device,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
    )

    # --- Initialize Model and Optimizer ---
    # --- 初始化模型和优化器 ---
    n_params["epoch_num"] = len(train_cf) // args.batch_size + 1
    model = M2GNN_word(n_params, args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Preparation ---
    # --- 训练准备 ---
    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    print("-------------------- Start Training ------------------------")

    for epoch in range(args.epoch):
        # --- Training Loop ---
        # --- 训练循环 ---
        loss, s = 0, 0
        train_s_t = time()
        model.train()
        for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
            batch_loss = model(input_nodes, blocks, pos_pair_graph, neg_pair_graph)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
        train_e_t = time()

        # --- Evaluation ---
        # --- 评估 ---
        if epoch % args.eval_interval == (args.eval_interval - 1):
            with torch.no_grad():
                model.eval()
                # Generate full embeddings for all users and items
                # 为所有用户和物品生成完整的嵌入
                for input_nodes, pos_pair_graph, blocks in test_dataloader:
                    model.generate(input_nodes, blocks, pos_pair_graph)
                user_emb = model.user_embed_final
                item_emb = model.item_embed_final

            test_s_t = time()
            ret = test_multi(user_emb, item_emb, user_dict, n_params, model)
            test_e_t = time()

            # --- Log Results ---
            # --- 记录结果 ---
            train_res = PrettyTable()
            train_res.field_names = [
                "Epoch", "Train Time", "Test Time", "Loss",
                "Recall", "NDCG", "Precision", "Hit Ratio"
            ]
            train_res.add_row([
                epoch, f"{train_e_t - train_s_t:.2f}s", f"{test_e_t - test_s_t:.2f}s", f"{loss:.4f}",
                ret["recall"], ret["ndcg"], ret["precision"], ret["hit_ratio"]
            ])
            print(train_res)

            # --- Early Stopping ---
            # --- 早停机制 ---
            cur_best_pre_0, stopping_step, should_stop = early_stopping(
                ret["recall"][0],
                cur_best_pre_0,
                stopping_step,
                expected_order="acc",
                flag_step=args.duration_epoch,
            )
            if should_stop:
                break

            # --- Save Model Checkpoint ---
            # --- 保存模型检查点 ---
            if ret["recall"][0] == cur_best_pre_0 and args.save_model:
                # The model save path is now constructed from args
                # 模型保存路径现在由参数构建
                save_dir = os.path.join(args.out_dir, args.dataset)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "model.ckpt")
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")

        else:
            print(f"Epoch {epoch}: Training Time: {train_e_t - train_s_t:.2f}s, Loss: {loss:.4f}")

    print(f"Early stopping at epoch {epoch}, best Recall@20: {cur_best_pre_0:.4f}")
    
    # The following command was OS-specific and has been removed for cross-platform compatibility.
    # It was originally intended to kill the training process.
    # 以下命令是特定于操作系统的，为了跨平台兼容性已被移除。
    # 它最初的目的是终止训练进程。
    # os.system("ps -ef | grep main_M2GNN | grep -v grep | cut -c 9-15 | xargs kill -9")
