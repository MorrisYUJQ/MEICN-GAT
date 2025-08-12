__author__ = "Jiaqian Yu"

import math
import random

# from torch_scatter import scatter_mean
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, HeteroGraphConv


import dgl.function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair

class M2GNN_word(nn.Module):
    """
    Top-level model that wraps the M2GNN_c core and handles loss calculation.
    顶层模型，封装了M2GNN_c核心并处理损失计算。
    """
    def __init__(self, data_config, args_config):
        super(M2GNN_word, self).__init__()

        self.n_users = data_config["n_users"]
        self.n_reviews = data_config["n_items"] # Note: This seems to be the number of review nodes, not items. / 注意：这似乎是评论节点的数量，而不是物品数量。
        self.n_items = data_config["n_items4rs"]
        self.n_tags = data_config["n_tag"]
        self.K_word2cf = 1.0 # Weight for the word loss component. / 词损失部分的权重。

        self.decay = args_config.l2
        self.lr = args_config.lr
        self.dim = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate

        self.iteration = args_config.iteration
        self.max_K = args_config.max_K
        self.max_len = args_config.max_len
        self.gamma = args_config.gamma

        self.device = torch.device(f"cuda:{args_config.gpu_id}" if torch.cuda.is_available() else "cpu")

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)

        self.gcn = self._init_model()

    def _init_weight(self):
        """
        Initialize embeddings and weights.
        初始化嵌入和权重。
        """
        initializer = nn.init.xavier_uniform_
        gain = 1.414

        # Combined embedding for users, reviews, and tags.
        # 用户、评论和标签的组合嵌入。
        self.all_embed = nn.Parameter(initializer(torch.empty(int(self.n_users + self.n_reviews + self.n_tags), self.dim), gain=gain))
        
        # Embeddings for negative sampling in word2vec loss.
        # 用于word2vec损失中负采样的嵌入。
        self.v_embeddings = nn.Embedding(self.n_tags, self.dim)
        nn.init.xavier_uniform_(self.v_embeddings.weight)

        # Final embeddings for evaluation. These are filled during the generate() step.
        # 用于评估的最终嵌入。在generate()步骤中填充。
        self.user_embed_final = nn.Parameter(torch.zeros(size=(self.n_users, self.dim)), requires_grad=False)
        self.item_embed_final = nn.Parameter(torch.zeros(size=(self.n_items, self.dim)), requires_grad=False)

    def _init_model(self):
        """
        Initialize the core GNN model.
        初始化核心GNN模型。
        """
        return M2GNN_c(
            dim=self.dim,
            n_hops=self.context_hops,
            n_users=self.n_users,
            n_reviews=self.n_reviews,
            n_tags=self.n_tags,
        )

    def forward(self, input_nodes, blocks, pos_pair_graph, neg_pair_graph):
        """
        Forward pass for training. Computes a combined loss.
        训练的前向传播。计算一个组合损失。
        """
        # Get embeddings from the GNN model.
        # 从GNN模型获取嵌入。
        user_gcn_emb, item_gcn_emb = self.gcn(blocks, input_nodes, self.all_embed, is_training=True)

        # --- BPR Loss Calculation ---
        # --- BPR损失计算 ---
        u_e = user_gcn_emb[pos_pair_graph.edges(etype="interaction")[0]]
        pos_e = item_gcn_emb[pos_pair_graph.edges(etype="interaction")[1], :]
        neg_e = item_gcn_emb[neg_pair_graph.edges(etype="interaction")[1], :]
        bpr_loss = self.create_bpr_loss(u_e, pos_e, neg_e)

        # --- Word2Vec-style Loss for Tags ---
        # --- 针对标签的Word2Vec风格损失 ---
        tag_embed = self.all_embed[self.n_users + self.n_reviews :, :]
        center_word_idx = pos_pair_graph.edges(etype="t2t")[0]
        pos_word_idx = pos_pair_graph.edges(etype="t2t")[1]
        neg_word_idx = neg_pair_graph.edges(etype="t2t")[1]
        
        tag_node_idx = pos_pair_graph.ndata[dgl.NID]["t"]
        
        emb_u = tag_embed[tag_node_idx[center_word_idx]]
        emb_v = self.v_embeddings(tag_node_idx[pos_word_idx])
        neg_emb_v = self.v_embeddings(tag_node_idx[neg_word_idx])
        word_loss = self.create_word_loss(emb_u, emb_v, neg_emb_v)

        return bpr_loss + self.K_word2cf * word_loss

    def generate(self, input_nodes, blocks, pos_pair_graph):
        """
        Generate user and item embeddings for evaluation.
        为评估生成用户和物品嵌入。
        """
        user_gcn_emb, item_gcn_emb = self.gcn(blocks, input_nodes, self.all_embed, is_training=False)
        
        u_e_idx = pos_pair_graph.ndata[dgl.NID]["h"]
        pos_e_idx = pos_pair_graph.ndata[dgl.NID]["u"]

        self.user_embed_final.data[u_e_idx] = user_gcn_emb.detach()
        self.item_embed_final.data[pos_e_idx] = item_gcn_emb.detach()

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        """
        Bayesian Personalized Ranking (BPR) loss.
        贝叶斯个性化排名（BPR）损失。
        """
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        
        # L2 regularization
        regularizer = (torch.norm(users)**2 + torch.norm(pos_items)**2 + torch.norm(neg_items)**2) / 2
        emb_loss = self.decay * regularizer / users.shape[0]
        
        return mf_loss + emb_loss

    def create_word_loss(self, emb_u, emb_v, neg_emb_v):
        """
        Word2Vec-style negative sampling loss for tag embeddings.
        针对标签嵌入的Word2Vec风格负采样损失。
        """
        if emb_u.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = F.logsigmoid(score)

        neg_score = torch.sum(torch.mul(emb_u, neg_emb_v), dim=1)
        neg_score = F.logsigmoid(-neg_score)

        loss_word = -1 * (torch.mean(score) + torch.mean(neg_score))
        return loss_word


class M2GNN_one_GAT(nn.Module):
    """
    A single layer of the MEICN-GAT model, incorporating GAT and Capsule Networks.
    MEICN-GAT模型的单层，集成了GAT和胶囊网络。
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(M2GNN_one_GAT, self).__init__()
        self.dim = dim
        
        # --- Capsule Network Parameters ---
        # --- 胶囊网络参数 ---
        self.num_low_capsules = 8
        self.num_mid_capsules = 4
        self.num_high_capsules = 2

        # --- Step 1: Low-level Expert Capsules ---
        # --- 步骤1：低级专家胶囊 ---
        # Each expert capsule processes one dimension of the input embedding.
        # 每个专家胶囊处理输入嵌入的一个维度。
        self.feature_linears = nn.ModuleList([nn.Linear(1, dim) for _ in range(self.num_low_capsules)])
        self.gate_networks = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())
            for _ in range(self.num_low_capsules)
        ])

        # --- Step 2 & 4: GAT Layers ---
        # --- 步骤2和4：GAT层 ---
        # GAT for processing heterogeneous graph neighborhoods.
        # 用于处理异构图邻域的GAT。
        self.gat_layers = HeteroGraphConv({
            'h_k_t': GATConv(dim, dim // num_heads, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout),
            'h_u_k_t': GATConv(dim, dim // num_heads, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout),
            't2t': GATConv(dim, dim // num_heads, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout),
            'u_k_t': GATConv(dim, dim // num_heads, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout),
            'interaction': GATConv(dim, dim // num_heads, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout),
            'test': GATConv(dim, dim // num_heads, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout),
        }, aggregate='sum')

        # GAT for processing the temporary graph of mid-level capsules.
        # 用于处理中级胶囊临时图的GAT。
        self.gat_mid_to_high = GATConv(dim, dim, num_heads, feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True)

        # --- Step 3 & 5: Dynamic Routing ---
        # --- 步骤3和5：动态路由 ---
        self.capsule_weights1 = nn.Parameter(torch.randn(self.num_mid_capsules, dim, dim))
        self.capsule_weights2 = nn.Parameter(torch.randn(self.num_high_capsules, dim, dim))
        nn.init.xavier_uniform_(self.capsule_weights1)
        nn.init.xavier_uniform_(self.capsule_weights2)
        
        # --- Step 6: Final Gating and Output ---
        # --- 步骤6：最终门控和输出 ---
        self.gate_network = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, self.num_high_capsules), nn.Sigmoid())
        self.output_projection = nn.Linear(self.num_high_capsules * dim, dim)

    def forward(self, block, user_embed, review_embed, tag_embed):
        """
        The data flow follows the architecture diagram (Figure 2 in the paper).
        数据流遵循架构图（论文中的图2）。
        """
        batch_size = user_embed.size(0)

        # --- Step 1: Low-level Expert Capsules ---
        # --- 步骤1：低级专家胶囊 ---
        # Disentangle user interests by forcing each capsule to focus on one feature dimension.
        # 通过强制每个胶囊关注一个特征维度来解耦用户兴趣。
        low_capsules = []
        for i in range(self.num_low_capsules):
            feature = user_embed[:, i].unsqueeze(1).float()
            feature_emb = self.feature_linears[i](feature)
            gate = self.gate_networks[i](feature_emb)
            capsule = gate * feature_emb
            low_capsules.append(capsule)
        low_capsules = torch.stack(low_capsules, dim=1) # [B, num_low_caps, D]

        # --- Step 2: GAT on Heterogeneous Graph ---
        # --- 步骤2：异构图上的GAT ---
        # Aggregate information from neighbors in the input graph.
        # 从输入图的邻居中聚合信息。
        h_dict = {'h': low_capsules.mean(dim=1), 'u': review_embed, 't': tag_embed}
        x = self.gat_layers(block, h_dict)
        
        # Extract the processed user, review, and tag embeddings.
        # 提取处理后的用户、评论和标签嵌入。
        user_embed_gat = x.get('h', user_embed).view(batch_size, -1)
        review_embed_gat = x.get('u', review_embed).view(review_embed.size(0), -1)
        tag_embed_gat = x.get('t', tag_embed).view(tag_embed.size(0), -1) if tag_embed.numel() > 0 else tag_embed
        
        # --- Step 3: Dynamic Routing to Mid-level Capsules ---
        # --- 步骤3：到中级胶囊的动态路由 ---
        # The GAT-processed user embedding is now treated as the input to the capsule routing.
        # 经过GAT处理的用户嵌入现在被视为胶囊路由的输入。
        input_caps = user_embed_gat.unsqueeze(1).expand(-1, self.num_low_capsules, -1)
        v1 = self.dynamic_routing(input_caps, self.capsule_weights1, self.num_mid_capsules)

        # --- Step 4: GAT on Temporary Capsule Graph ---
        # --- 步骤4：临时胶囊图上的GAT ---
        # Create a temporary fully-connected graph for mid-level capsules and apply GAT.
        # 为中级胶囊创建一个临时的全连接图并应用GAT。
        v1_flat = v1.reshape(-1, self.dim)
        num_nodes = self.num_mid_capsules
        edges_src = torch.repeat_interleave(torch.arange(num_nodes), num_nodes - 1)
        edges_dst = torch.cat([torch.cat((torch.arange(i), torch.arange(i+1, num_nodes))) for i in range(num_nodes)])
        
        graphs = [dgl.graph((edges_src, edges_dst), num_nodes=num_nodes) for _ in range(batch_size)]
        batched_graph = dgl.batch(graphs).to(user_embed.device)
        v1_gat = self.gat_mid_to_high(batched_graph, v1_flat).view(batch_size, self.num_mid_capsules, -1)
        
        # --- Step 5: Dynamic Routing to High-level Capsules ---
        # --- 步骤5：到高级胶囊的动态路由 ---
        v2 = self.dynamic_routing(v1_gat, self.capsule_weights2, self.num_high_capsules)

        # --- Step 6: Gating and Final Output ---
        # --- 步骤6：门控和最终输出 ---
        activation_probs = self.gate_network(user_embed_gat).unsqueeze(-1)
        v2_gated = v2 * activation_probs
        
        capsule_features = v2_gated.reshape(batch_size, -1)
        output = self.output_projection(capsule_features)

        return output, review_embed_gat, tag_embed_gat

    def dynamic_routing(self, u_hat_in, weights, num_out_caps, iterations=3):
        """
        Performs the dynamic routing algorithm.
        执行动态路由算法。
        """
        batch_size, num_in_caps, _ = u_hat_in.shape
        u_hat = torch.einsum('bnd,mdo->bnmo', u_hat_in, weights)
        
        b = torch.zeros(batch_size, num_in_caps, num_out_caps, device=u_hat_in.device)
        
        for i in range(iterations):
            c = F.softmax(b, dim=2).unsqueeze(-1)
            s = (c * u_hat).sum(dim=1)
            v = self.squash(s)
            if i < iterations - 1:
                agreement = torch.sum(u_hat * v.unsqueeze(1), dim=-1)
                b = b + agreement
        return v

    def squash(self, x, epsilon=1e-8):
        """
        The non-linear activation function used in Capsule Networks.
        胶囊网络中使用的非线性激活函数。
        """
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / (torch.sqrt(squared_norm) + epsilon)



class M2GNN_c(nn.Module):
    """
    The core GNN model, which stacks multiple layers of M2GNN_one_GAT.
    核心GNN模型，它堆叠了多层M2GNN_one_GAT。
    """
    def __init__(
        self,
        dim,
        n_hops,
        n_users,
        n_reviews,
        n_tags,
    ):
        super(M2GNN_c, self).__init__()
        self.dim = dim
        self.n_hops = n_hops
        self.n_users = n_users
        self.n_reviews = n_reviews
        self.n_tags = n_tags
        
        self.convs = nn.ModuleList()
        for _ in range(n_hops):
            self.convs.append(
                M2GNN_one_GAT(dim=dim)
            )
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, blocks, input_nodes, all_embed, is_training):
        """
        Forward pass through the stacked GNN layers.
        通过堆叠的GNN层进行前向传播。
        """
        # Initial node embeddings from the input feature tensor.
        # 从输入特征张量中获取初始节点嵌入。
        user_embed = all_embed[: self.n_users, :][input_nodes["h"], :]
        review_embed = all_embed[self.n_users : self.n_users + self.n_reviews, :][input_nodes["u"], :]
        tag_embed = all_embed[self.n_users + self.n_reviews :, :][input_nodes["t"], :]
        
        if is_training:
            user_embed = self.dropout(user_embed)
            review_embed = self.dropout(review_embed)
            tag_embed = self.dropout(tag_embed)

        # Propagate through the GNN layers.
        # 通过GNN层进行传播。
        for i in range(self.n_hops):
            user_embed, review_embed, tag_embed = self.convs[i](
                blocks[i],
                user_embed,
                review_embed,
                tag_embed
            )

            if is_training and i < self.n_hops - 1: # No dropout on the last layer's output
                user_embed = self.dropout(user_embed)
                review_embed = self.dropout(review_embed)
                if tag_embed.numel() > 0:
                    tag_embed = self.dropout(tag_embed)

        # Return the final embeddings for the destination nodes in the last block.
        # 返回最后一个块中目标节点的最终嵌入。
        num_review_dst = blocks[-1].num_dst_nodes("u")
        num_user_dst = blocks[-1].num_dst_nodes("h")
        
        return user_embed[:num_user_dst, :], review_embed[:num_review_dst, :]




