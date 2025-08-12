import argparse


def parse_args():
    """
    Parse command line arguments for the MEICN-GAT model.
    为MEICN-GAT模型解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="MEICN-GAT for High-end Business Markets")

    # --- Dataset Settings ---
    # --- 数据集设置 ---
    parser.add_argument('--dataset', type=str, default='hotel', help='Choose a dataset: [hotel, gift_card, etc.]')
    parser.add_argument('--data_path', type=str, default='./data/processed/', help='Input data path.')

    # --- Training Settings ---
    # --- 训练设置 ---
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--test_batch_size', type=int, default=4096, help='Batch size for testing.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--l2', type=float, default=1e-5, help='L2 regularization weight.')
    parser.add_argument('--dim', type=int, default=16, help='Embedding dimension.')

    # --- Model Architecture Settings ---
    # --- 模型架构设置 ---
    parser.add_argument('--context_hops', type=int, default=2, help='Number of context hops in GNN.')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum number of neighbors to sample.')
    parser.add_argument('--iteration', type=int, default=3, help='Number of routing iterations in capsule network.')

    # --- Evaluation and Early Stopping ---
    # --- 评估与早停设置 ---
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluate on the test set every N epochs.')
    parser.add_argument('--duration_epoch', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--Ks', nargs='?', default='[20, 50, 100]', help='K values for evaluation metrics (e.g., Recall@K).')

    # --- GPU Settings ---
    # --- GPU设置 ---
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use.')

    # --- Save and Load Settings ---
    # --- 保存与加载设置 ---
    parser.add_argument('--save_model', action='store_true', help='Flag to save the best model.')
    parser.add_argument('--out_dir', type=str, default='./checkpoints/', help='Output directory for model checkpoints.')
    
    return parser.parse_args()
