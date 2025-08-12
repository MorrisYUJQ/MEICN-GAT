import json
import os
import pickle
import dgl

def preprocess_and_save(input_dir, output_dir, train_split=0.8):
    """
    Load raw data, preprocess it, build a heterogeneous graph, and save the processed data.
    加载原始数据，进行预处理，构建异构图，并保存处理后的数据。

    Args:
        input_dir (str): Directory containing the raw JSON files ('processed_hotel_comments1.json', 'merged_tag_results.json').
                         包含原始JSON文件（'processed_hotel_comments1.json', 'merged_tag_results.json'）的目录。
        output_dir (str): Directory to save the processed graph and train/test data.
                          用于保存处理后的图和训练/测试数据的目录。
        train_split (float): The proportion of data to use for the training set.
                             用于训练集的数据比例。
    """
    
    # --- 1. Load Raw Data and Create ID Mappings ---
    # --- 1. 加载原始数据并创建ID映射 ---
    
    print("Loading raw data...")
    with open(os.path.join(input_dir, "processed_hotel_comments1.json"), 'r', encoding='utf-8') as f:
        hotel_comments = json.load(f)
    with open(os.path.join(input_dir, "merged_tag_results.json"), 'r', encoding='utf-8') as f:
        tag_results = json.load(f)

    # Create mappings from original IDs to sequential integer IDs
    # 创建从原始ID到连续整数ID的映射
    hotel_id_map = {hotel_id: i for i, hotel_id in enumerate(set(c["hotelId"] for c in hotel_comments))}
    user_id_map = {user_id: i for i, user_id in enumerate(set(c["userid"] for c in hotel_comments))}
    tag_id_map = {tag_id: i for i, tag_id in enumerate(set(t["tag_id"] for t in tag_results))}

    # --- 2. Build Relationships Based on Meta-paths ---
    # --- 2. 基于元路径构建关系 ---
    
    print("Building relationships from meta-paths...")
    # Keyword -> tag mapping
    keyword_to_tag = {kw: tag_id_map[t["tag_id"]] for t in tag_results for kw in t["tags"]}
    
    # User -> keyword counts
    user_keywords = {}
    for comment in hotel_comments:
        uid = user_id_map[comment["userid"]]
        user_keywords.setdefault(uid, {})
        for kw in comment["keywords"]:
            user_keywords[uid][kw] = user_keywords[uid].get(kw, 0) + 1
            
    # Hotel -> keyword counts
    hotel_keywords = {}
    for comment in hotel_comments:
        hid = hotel_id_map[comment["hotelId"]]
        hotel_keywords.setdefault(hid, {})
        for kw in comment["keywords"]:
            hotel_keywords[hid][kw] = hotel_keywords[hid].get(kw, 0) + 1

    # (u, t) edges from "u-k-t" meta-path
    u_t_edges = {}
    for uid, keywords in user_keywords.items():
        u_t_edges.setdefault(uid, {})
        for kw, count in keywords.items():
            if kw in keyword_to_tag:
                tid = keyword_to_tag[kw]
                u_t_edges[uid][tid] = u_t_edges[uid].get(tid, 0) + count

    # (h, t) edges from "h-k-t" meta-path
    h_t_edges = {}
    for hid, keywords in hotel_keywords.items():
        h_t_edges.setdefault(hid, {})
        for kw, count in keywords.items():
            if kw in keyword_to_tag:
                tid = keyword_to_tag[kw]
                h_t_edges[hid][tid] = h_t_edges[hid].get(tid, 0) + count

    # (t, t) edges from sentiment consistency
    tag_sentiments = {tag_id_map[t["tag_id"]]: t["sentiment"] for t in tag_results}
    t_t_edges = []
    for t1, s1 in tag_sentiments.items():
        for t2, s2 in tag_sentiments.items():
            if t1 != t2 and s1 == s2:
                t_t_edges.append((t1, t2))

    # Split interactions into train and test sets
    # 将交互数据分割为训练集和测试集
    h_u_train, h_u_test = [], []
    train_user_set, test_user_set = {}, {}
    split_idx = int(len(hotel_comments) * train_split)
    
    for i, comment in enumerate(hotel_comments):
        hid = hotel_id_map[comment["hotelId"]]
        uid = user_id_map[comment["userid"]]
        if i < split_idx:
            h_u_train.append((hid, uid))
            train_user_set.setdefault(hid, []).append(uid)
        else:
            h_u_test.append((hid, uid))
            test_user_set.setdefault(hid, []).append(uid)

    # (h, t) edges from "h-u-k-t" meta-path (collaborative signal)
    h_u_k_t_edges = {}
    for hid, uids in train_user_set.items():
        h_u_k_t_edges.setdefault(hid, {})
        for uid in uids:
            if uid in u_t_edges:
                for tid, count in u_t_edges[uid].items():
                    h_u_k_t_edges[hid][tid] = h_u_k_t_edges[hid].get(tid, 0) + count

    # --- 3. Construct and Save DGL Graph and Data ---
    # --- 3. 构建并保存DGL图和数据 ---

    print("Constructing DGL graph...")
    graph_data = {
        ("t", "h_k_t", "h"): ([t for h, ts in h_t_edges.items() for t in ts.keys()],
                             [h for h, ts in h_t_edges.items() for t in ts.keys()]),
        ("t", "h_u_k_t", "h"): ([t for h, ts in h_u_k_t_edges.items() for t in ts.keys()],
                               [h for h, ts in h_u_k_t_edges.items() for t in ts.keys()]),
        ("t", "t2t", "t"): ([e[0] for e in t_t_edges], [e[1] for e in t_t_edges]),
        ("t", "u_k_t", "u"): ([t for u, ts in u_t_edges.items() for t in ts.keys()],
                             [u for u, ts in u_t_edges.items() for t in ts.keys()]),
        ("h", "interaction", "u"): ([e[0] for e in h_u_train], [e[1] for e in h_u_train]),
        ("h", "test", "u"): ([e[0] for e in h_u_test], [e[1] for e in h_u_test]),
    }
    
    num_nodes_dict = {'h': len(hotel_id_map), 'u': len(user_id_map), 't': len(tag_id_map)}
    graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

    print("\nGraph Construction Summary:")
    print(f"Node types: {graph.ntypes}")
    print(f"Edge types: {graph.etypes}")
    for ntype in graph.ntypes:
        print(f"Number of '{ntype}' nodes: {graph.num_nodes(ntype)}")
    for etype in graph.etypes:
        print(f"Number of '{etype}' edges: {graph.num_edges(etype)}")

    # Save processed data
    # 保存处理后的数据
    os.makedirs(output_dir, exist_ok=True)
    
    # Save graph object (using a single object for simplicity)
    # 保存图对象（为简单起见使用单个对象）
    graph_output_path = os.path.join(output_dir, "graph_tag.pkl")
    with open(graph_output_path, "wb") as f:
        pickle.dump(None, f) # Placeholder for graph_dp if needed
        pickle.dump(graph, f)
    print(f"\nGraph saved to {graph_output_path}")
    
    # Save train/test sets and mappings
    # 保存训练/测试集和映射
    data_output_path = os.path.join(output_dir, "train_test.pkl")
    with open(data_output_path, "wb") as f:
        pickle.dump(h_u_train, f)
        pickle.dump(h_u_test, f)
        pickle.dump(len(user_id_map), f) # len_item seems to be total user count
        pickle.dump(train_user_set, f)
        pickle.dump(test_user_set, f)
    print(f"Train/test data saved to {data_output_path}")

if __name__ == '__main__':
    # --- Example Usage ---
    # --- 使用示例 ---
    # Define your input and output directories
    # 定义你的输入和输出目录
    # Note: You should place your 'processed_hotel_comments1.json' and 'merged_tag_results.json'
    # in the 'raw_data' directory before running.
    # 注意：在运行前，您应该将'processed_hotel_comments1.json'和'merged_tag_results.json'
    # 放入'raw_data'目录中。
    
    # Create dummy raw_data folder for demonstration
    if not os.path.exists("./raw_data"):
        os.makedirs("./raw_data")
        print("Created a dummy './raw_data' directory. Please place your JSON files here.")
    
    INPUT_DATA_DIR = "./raw_data"
    OUTPUT_DATA_DIR = "./processed_data/hotel" # Example for 'hotel' dataset
    
    preprocess_and_save(INPUT_DATA_DIR, OUTPUT_DATA_DIR)
