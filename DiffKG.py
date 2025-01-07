r"""
DiffKG
##################################################
Reference:
    Yangqin Jiang et al. "DiffKG: Knowledge Graph Diffusion Model for Recommendation." in WSDM 2024 Oral.

Reference code:
   https://github.com/HKUDS/DiffKG?tab=readme-ov-file
"""
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from recbole.data.interaction import Interaction
import torch.cuda.amp as amp
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_softmax
from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from logging import getLogger
from pickletools import optimize
from time import time
from turtle import forward
import torch.utils.data as data
from recbole.data.dataloader.general_dataloader import FullSortEvalDataLoader
from recbole.evaluator import Evaluator
from recbole.evaluator.collector import Collector
import math
from collections import defaultdict
from recbole.utils import (
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    EvaluatorType,
    KGDataLoaderState,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger,
)
import os
from torch.nn.parallel import DistributedDataParallel
init = nn.init.xavier_uniform_

class RGAT(nn.Module):
    def __init__(self, channel, n_hops,res_lambda, mess_dropout_rate=0.4):
        super(RGAT, self).__init__()
        self.mess_dropout_rate = mess_dropout_rate
        self.W = nn.Parameter(init(torch.empty(size=(2*channel, channel)), gain=nn.init.calculate_gain('relu')))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.n_hops = n_hops
        self.dropout = nn.Dropout(p=mess_dropout_rate)

        self.res_lambda = res_lambda

    def agg(self, entity_emb, relation_emb, kg):
        edge_index, edge_type = kg
        head, tail = edge_index
        a_input = torch.cat([entity_emb[head], entity_emb[tail]], dim=-1)
        e_input = torch.multiply(torch.mm(a_input, self.W), relation_emb[edge_type]).sum(-1)
        e = self.leakyrelu(e_input)
        e = scatter_softmax(e, head, dim=0, dim_size=entity_emb.shape[0])
        agg_emb = entity_emb[tail] * e.view(-1, 1)
        agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])
        agg_emb = agg_emb + entity_emb
        return agg_emb
    
    def forward(self, entity_emb, relation_emb, kg, mess_dropout=True):
        entity_emb = entity_emb.weight
        entity_res_emb = entity_emb
        for _ in range(self.n_hops):
            entity_emb = self.agg(entity_emb, relation_emb.weight, kg)
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)

            entity_res_emb = self.res_lambda * entity_res_emb + entity_emb
        return entity_res_emb
    
class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size,device, norm=False, dropout=0.5):
        super(Denoise, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm
        self.device = device

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        seed = 2020
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, mess_dropout=True):
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).to(self.device)
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        if mess_dropout:
            x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h

class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps,item_num,e_loss, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()

        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.item_num = item_num
        self.e_loss = e_loss

        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
            if beta_fixed:
                self.betas[0] = 0.0001

            self.calculate_for_diffusion()

    def get_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
        return np.array(betas)
    
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

    def p_sample(self, model, x_start, steps):
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
            x_t = self.q_sample(x_start, t)
        
        indices = list(range(self.steps))[::-1]

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).cuda()
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
            x_t = model_mean
        return x_t
            
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.cuda()
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def p_mean_variance(self, model, x, t):
        model_output = model(x, t, False)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
        
        return model_mean, model_log_variance

    def cal_loss_diff(self, model, batch, ui_matrix, userEmbeds, itmEmbeds):
        x_start, batch_index = batch

        batch_size = x_start.size(0)
        ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        model_output = model(x_t, ts)
        mse = self.mean_flat((x_start - model_output) ** 2)

        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)

        diff_loss = weight * mse

        item_user_matrix = torch.spmm(ui_matrix, model_output[:, :self.item_num].t()).t()
        itmEmbeds_kg = torch.mm(item_user_matrix, userEmbeds)
        ukgc_loss = self.mean_flat((itmEmbeds_kg - itmEmbeds[batch_index]) ** 2)

        loss = diff_loss.mean() * (1 - self.e_loss) + ukgc_loss.mean() * self.e_loss
        losses = {'diff loss': diff_loss, 'ukgc loss': ukgc_loss}
        return loss, losses
        
    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.cuda()
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])


class diffkg(KnowledgeRecommender):
    input_type = InputType.PAIRWISE
    
    def __init__(self, config, dataset):
        super(diffkg, self).__init__(config, dataset)
        self.embedding_size = config["embedding_size"]
        self.layer_num = config["layer_num"]
        self.layer_num_kg = config['layer_num_kg']
        self.mess_dropout_rate = config['mess_dropout_rate']
        self.triplet_num = config['triplet_num']
        self.cl_pattern = config['cl_pattern']
        self.sampling_steps = config['sampling_steps']
        self.rebuild_k = config['rebuild_k']
        self.keepRate = config['keepRate']
        self.res_lambda = config['res_lambda']
        self.noise_scale = config['noise_scale']
        self.noise_min = config['noise_min']
        self.noise_max = config['noise_max']
        self.steps = config['steps']
        self.e_loss = config['e_loss']
        self.d_emb_size = config['d_emb_size']
        self.reg_weight = config['reg_weight']
        self.temperature = config['temperature']
        self.cl_weight = config['cl_weight']
        seed = 2020
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.uEmbeds = nn.Embedding(self.n_users, self.embedding_size).to(self.device)
        self.eEmbeds = nn.Embedding(self.n_entities, self.embedding_size).to(self.device)
        self.rEmbeds = nn.Embedding(self.n_relations*2 -2, self.embedding_size).to(self.device)
        # self.uEmbeds = nn.Parameter(init(torch.empty(self.n_users, self.embedding_size))).to(self.device)
        # self.eEmbeds = nn.Parameter(init(torch.empty(self.n_entities, self.embedding_size))).to(self.device)
        # self.rEmbeds = nn.Parameter(init(torch.empty(self.n_relations*2 -2, self.embedding_size))).to(self.device)
        # self.uEmbeds = nn.Parameter(torch.randn(self.n_users, self.embedding_size)).to(self.device)
        # self.eEmbeds = nn.Parameter(torch.randn(self.n_entities, self.embedding_size)).to(self.device)
        # self.rEmbeds = nn.Parameter(torch.randn(self.n_relations * 2, self.embedding_size)).to(self.device)

        #self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(self.GNN_layers)])

        self.rgat = RGAT(self.embedding_size, self.layer_num_kg,self.res_lambda, self.mess_dropout_rate).to(self.device)

        self.kg_dict = dataset.kg_graph(
            form="coo", value_field="relation_id"
        )
        self.build_kg_dict()
        self.kg_edges,self.kg_dict  = self._build_graphs_diff(self.kg_dict)
        self.edge_index, self.edge_type = self._sample_edges_from_dict(self.kg_dict, triplet_num=self.triplet_num)
  
        adj = dataset.inter_matrix(form="coo").astype(np.float32)
        self.adj = self._make_torch_adj(adj)
        
        self.adj_a = dataset.inter_matrix(form="coo").astype(np.float32)
        indices = np.vstack((self.adj_a.row, self.adj_a.col))
        values = self.adj_a.data
        self.adj_a = torch.sparse_coo_tensor(indices, values, size=self.adj_a.shape).to(self.device)
        self.test = 1
        self.apply(xavier_normal_initialization)
        
    def _normalize_adj(self, mat):
        """Laplacian normalization for mat in coo_matrix

        Args:
            mat (scipy.sparse.coo_matrix): the un-normalized adjacent matrix

        Returns:
            scipy.sparse.coo_matrix: normalized adjacent matrix
        """
        # Add epsilon to avoid divide by zero
        degree = np.array(mat.sum(axis=-1)) + 1e-10
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()
        
    def _make_torch_adj(self, mat):
        """Transform uni-directional adjacent matrix in coo_matrix into bi-directional adjacent matrix in torch.sparse.FloatTensor

        Args:
        mat (coo_matrix): the uni-directional adjacent matrix

        Returns:
        torch.sparse.FloatTensor: the bi-directional matrix
        """
        a = csr_matrix((self.n_users, self.n_users))
        b = csr_matrix((self.n_items, self.n_items))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0# MARK
        mat = self._normalize_adj(mat)

        # make torch tensor
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).to(self.device)
        
        
    def _build_graphs_diff(self, triplets):
        kg_dict = defaultdict(list)
        # h, t, r
        kg_edges = list()
        
        print("Begin to load knowledge graph triples ...")
        
        kg_counter_dict = {}
        
        for h_id, r_id, t_id in tqdm(triplets, ascii=True):
            if h_id not in kg_counter_dict.keys():
                kg_counter_dict[h_id] = set()
            if t_id not in kg_counter_dict[h_id]:
                kg_counter_dict[h_id].add(t_id)
            else:
                continue
            kg_edges.append([h_id, t_id, r_id])
            kg_dict[h_id].append((r_id, t_id))
            
        return kg_edges, kg_dict
            
    def build_kg_dict(self):
        indices = [self.kg_dict.row,self.kg_dict.col]
        values = list(self.kg_dict.data)

        # 初始化三元组列表
        triplets = []

        # 遍历非零元素并存储为三元组
        for i in range(len(indices[0])):
            row = indices[0][i]
            col = indices[1][i]
            value = values[i]
            triplets.append([row, value,col])
        can_triplets_np = np.array(triplets)
        can_triplets_np = np.unique(can_triplets_np, axis=0)
        
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        self.kg_dict = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)

  
    def _sample_edges_from_dict(self, kg_dict, triplet_num=None):
        sampleEdges = []
        for h in kg_dict:
            t_list = kg_dict[h]
            if triplet_num != -1 and len(t_list) > triplet_num:
                sample_edges_i = random.sample(t_list, triplet_num)
            else:
                sample_edges_i = t_list
            for r, t in sample_edges_i:
                sampleEdges.append([h, t, r])
        return self._get_edges(sampleEdges)
    
    def getEntityEmbeds(self):
        return self.eEmbeds.weight
    
    def getUserEmbeds(self):
        return self.uEmbeds.weight
    
    def setDenoisedKG(self, denoisedKG):
        self.denoisedKG = denoisedKG

    def _get_edges(self, kg_edges):
        graph_tensor = torch.tensor(kg_edges)
        index = graph_tensor[:, :-1]
        type = graph_tensor[:, -1]
        return index.t().long().to(self.device), type.long().to(self.device)
    
    def _pick_embeds(self, user_embeds, item_embeds, interaction):
        ancs, poss, negs = interaction['user_id'],interaction['item_id'],interaction['neg_item_id']
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds
    
    def reg_params(model):
        reg_loss = 0
        for W in model.parameters():
            reg_loss += W.norm(2).square()
        return reg_loss
    
    def cal_infonce_loss(self,embeds1, embeds2, all_embeds2, temp=1.0):
        """ InfoNCE Loss
        """
        normed_embeds1 = embeds1 / torch.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
        normed_embeds2 = embeds2 / torch.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
        normed_all_embeds2 = all_embeds2 / torch.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
        nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
        deno_term = torch.log(torch.sum(torch.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
        cl_loss = (nume_term + deno_term).sum()
        return cl_loss
    
    def _propagate(self, adj, embeds):
        return torch.spmm(adj, embeds)
    
    def forward(self, adj, mess_dropout=True, kg=None):
        if kg == None:
            hids_KG = self.rgat.forward(self.eEmbeds, self.rEmbeds, [self.edge_index, self.edge_type], mess_dropout)
        else:
            hids_KG = self.rgat.forward(self.eEmbeds, self.rEmbeds, kg, mess_dropout)
        
        embeds = torch.concat([self.uEmbeds.weight, hids_KG[:self.n_items, :]], axis=0)
        embedsLst = [embeds]
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embedsLst[-1])
            embedsLst.append(embeds)
        embeds = sum(embedsLst)
        return embeds[:self.n_users], embeds[self.n_users:]
    
    def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
        pos_preds = (anc_embeds * pos_embeds).sum(-1)
        neg_preds = (anc_embeds * neg_embeds).sum(-1)
        return torch.sum(F.softplus(neg_preds - pos_preds))
    
    def calculate_loss(self, interaction,denoisedKG):
        self.test = 1
        if self.cl_pattern == 0:
            user_embeds, item_embeds = self.forward(self.adj, kg=denoisedKG)
            user_embeds_kg, item_embeds_kg = self.forward(self.adj)
        else:
            user_embeds, item_embeds = self.forward(self.adj)
            user_embeds_kg, item_embeds_kg = self.forward(self.adj, kg=denoisedKG)
        
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, interaction)
        
        pos_preds = (anc_embeds * pos_embeds).sum(-1)
        neg_preds = (anc_embeds * neg_embeds).sum(-1)
        bpr_loss = torch.sum(F.softplus(neg_preds - pos_preds)) / anc_embeds.shape[0]
        # bpr_loss = self.cal_bpr_loss(anc_embeds = anc_embeds, pos_embeds = pos_embeds, neg_embeds = neg_embeds) / anc_embeds.shape[0]
        reg_loss = 0
        for W in self.parameters():
            reg_loss += W.norm(2).square()
        reg_loss = self.reg_weight * reg_loss

        anc_embeds_kg, pos_embeds_kg, neg_embeds_kg = self._pick_embeds(user_embeds_kg, item_embeds_kg, interaction)
        cl_loss = self.cal_infonce_loss(anc_embeds, anc_embeds_kg, user_embeds_kg, self.temperature) + self.cal_infonce_loss(pos_embeds, pos_embeds_kg, item_embeds_kg, self.temperature)
        cl_loss /= anc_embeds.shape[0]
        cl_loss *= self.cl_weight

        loss = bpr_loss + reg_loss + cl_loss
        
        return loss
    
    def _mask_predict(self, full_preds, train_mask):
        return full_preds * (1 - train_mask) - 1e8 * train_mask
    
    def full_sort_predict(self, interaction):
        if self.test ==1:
            if self.cl_pattern == 0:
                user_embeds, item_embeds = self.forward(self.adj, kg=self.denoisedKG)
            else:
                user_embeds, item_embeds = self.forward(self.adj)
            self.user_embeds, self.item_embeds = user_embeds, item_embeds
            self.test = 0
        user = interaction[self.USER_ID]
        u_emb = self.user_embeds[user]
        pck_mask = self.adj_a.index_select(index=user,dim=0).to_dense()
        pck_mask = pck_mask.view(-1)
        scores = torch.matmul(u_emb, self.item_embeds.transpose(0, 1))
        # full_preds = self._mask_predict(scores, pck_mask)

        return scores.view(-1)
    
    

class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model
        if not config["single_spec"]:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.distributed_model = DistributedDataParallel(
                self.model, device_ids=[config["local_rank"]]
            )

    def fit(self, train_data):
        r"""Train the model based on the train data."""
        raise NotImplementedError("Method [next] should be implemented.")

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data."""

        raise NotImplementedError("Method [next] should be implemented.")

    def set_reduce_hook(self):
        r"""Call the forward function of 'distributed_model' to apply grads
        reduce hook to each parameter of its module.

        """
        t = self.model.forward
        self.model.forward = lambda x: x
        self.distributed_model(torch.LongTensor([0]).to(self.device))
        self.model.forward = t

    def sync_grad_loss(self):
        r"""Ensure that each parameter appears to the loss function to
        make the grads reduce sync in each node.

        """
        sync_loss = 0
        for params in self.model.parameters():
            sync_loss += torch.sum(params) * 0
        return sync_loss


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.wandblogger = WandbLogger(config)
        self.learner = config["learner"]
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.eval_step = min(config["eval_step"], self.epochs)
        self.stopping_step = config["stopping_step"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.valid_metric = config["valid_metric"].lower()
        self.valid_metric_bigger = config["valid_metric_bigger"]
        self.test_batch_size = config["eval_batch_size"]
        self.gpu_available = torch.cuda.is_available() and config["use_gpu"]
        self.device = config["device"]
        print("self_device:",self.device)
        self.checkpoint_dir = config["checkpoint_dir"]
        self.enable_amp = config["enable_amp"]
        self.enable_scaler = torch.cuda.is_available() and config["enable_scaler"]
        ensure_dir(self.checkpoint_dir)
        saved_model_file = "{}-{}.pth".format(self.config["model"], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config["weight_decay"]

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.eval_type = config["eval_type"]
        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_tensor = None
        self.tot_item_num = None
        self.diffusion = GaussianDiffusion(self.model.noise_scale, self.model.noise_min, self.model.noise_max, self.model.steps,self.model.n_items,self.model.e_loss).cuda()
        out_dims = eval(config['dims']) + [self.model.n_entities]
        in_dims = out_dims[::-1]
        self.denoise = Denoise(in_dims, out_dims, self.model.d_emb_size,device=self.model.device, norm=True).cuda()
        self.kg_matrix = self.buildKGMatrix(self.model.kg_edges)
        # seed = 2020
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        # np.random.seed(seed)
        # random.seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        self.diffusionData = DiffusionData(self.kg_matrix.A)
        self.diffusionLoader = data.DataLoader(self.diffusionData, batch_size=config['train_batch_size'], shuffle=True, num_workers=0)
        # self.kg_edges, self.kg_dict = self._build_graphs_diff(self.model.kg_dict)
        self.relation_dict = self.RelationDictBuild()
        self.optimizer_denoise = optim.Adam(self.denoise.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.graph_tensor = None
        #self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

        
    def buildKGMatrix(self, kg_edges):
        edge_list = []
        for h_id, t_id, r_id in kg_edges:
            edge_list.append((h_id, t_id))
        edge_list = np.array(edge_list)
        
        kgMatrix = sp.csr_matrix((np.ones_like(edge_list[:,0]), (edge_list[:,0], edge_list[:,1])), dtype='float64', shape=(self.model.n_entities, self.model.n_entities))
        
        return kgMatrix
        
    def _build_graphs_diff(self, triplets):
        kg_dict = defaultdict(list)
        # h, t, r
        kg_edges = list()
        
        print("Begin to load knowledge graph triples ...")
        
        kg_counter_dict = {}
        
        for h_id, r_id, t_id in tqdm(triplets, ascii=True):
            if h_id not in kg_counter_dict.keys():
                kg_counter_dict[h_id] = set()
            if t_id not in kg_counter_dict[h_id]:
                kg_counter_dict[h_id].add(t_id)
            else:
                continue
            kg_edges.append([h_id, t_id, r_id])
            kg_dict[h_id].append((r_id, t_id))
            
        return kg_edges, kg_dict
        
    def RelationDictBuild(self):
        relation_dict = {}
        for head in self.model.kg_dict:
            relation_dict[head] = {}
            for (relation, tail) in self.model.kg_dict[head]:
                relation_dict[head][tail] = relation
        return relation_dict

    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop("params", self.model.parameters())
        learner = kwargs.pop("learner", self.learner)
        learning_rate = kwargs.pop("learning_rate", self.learning_rate)
        weight_decay = kwargs.pop("weight_decay", self.weight_decay)

        if (
            self.config["reg_weight"]
            and weight_decay
            and weight_decay * self.config["reg_weight"] > 0
        ):
            self.logger.warning(
                "The parameters [weight_decay] and [reg_weight] are specified simultaneously, "
                "which may lead to double regularization."
            )

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "sparse_adam":
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning(
                    "Sparse Adam cannot argument received argument [{weight_decay}]"
                )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        # train_data.dataset.sample_negs()
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )
        diffusionLoader = self.diffusionLoader
        ######################################################################
        loss_log_dict = {}
        ep_loss = 0
        self.model.train()
        # seed = 2020
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        # np.random.seed(seed)
        # random.seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        for _, tem in tqdm(enumerate(diffusionLoader), desc='Training Diffusion', total=len(diffusionLoader)):
            batch_data = list(map(lambda x: x.to(self.model.device), tem))

            ui_matrix = self.model.adj_a
            iEmbeds = self.model.getEntityEmbeds().detach()
            uEmbeds = self.model.getUserEmbeds().detach()

            self.optimizer_denoise.zero_grad()
            loss_diff, loss_dict_diff = self.diffusion.cal_loss_diff(self.denoise, batch_data, ui_matrix, uEmbeds, iEmbeds)
            loss_diff.backward()
            self.optimizer_denoise.step()

        with torch.no_grad():
            denoised_edges = []
            h_list = []
            t_list = []
            
            for _, tem in enumerate(diffusionLoader):
                batch_data = list(map(lambda x: x.to(self.model.device), tem))
                batch_item, batch_index = batch_data
                denoised_batch = self.diffusion.p_sample(self.denoise, batch_item, self.model.sampling_steps)
                top_item, indices_ = torch.topk(denoised_batch, k=self.model.rebuild_k)
                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]):
                        h_list.append(batch_index[i])
                        t_list.append(indices_[i][j])

            edge_set = set()
            for index in range(len(h_list)):
                edge_set.add((int(h_list[index].cpu().numpy()), int(t_list[index].cpu().numpy())))
            for index in range(len(h_list)):
                if (int(t_list[index].cpu().numpy()), int(h_list[index].cpu().numpy())) not in edge_set:
                    h_list.append(t_list[index])
                    t_list.append(h_list[index])
            relation_dict = self.relation_dict
            for index in range(len(h_list)):
                try:
                    denoised_edges.append([h_list[index], t_list[index], relation_dict[int(h_list[index].cpu().numpy())][int(t_list[index].cpu().numpy())]])
                except Exception:
                    continue
            try:
                graph_tensor = torch.tensor(denoised_edges)
                index_ = graph_tensor[:, :-1]
                type_ = graph_tensor[:, -1]
                denoisedKG = (index_.t().long().cuda(), type_.long().cuda())
                self.model.setDenoisedKG(denoisedKG)
                self.graph_tensor = graph_tensor
            except Exception:
                graph_tensor = self.graph_tensor
                index_ = graph_tensor[:, :-1]
                type_ = graph_tensor[:, -1]
                denoisedKG = (index_.t().long().cuda(), type_.long().cuda())
                self.model.setDenoisedKG(denoisedKG)
            # graph_tensor = torch.tensor(denoised_edges)
            # index_ = graph_tensor[:, :-1]
            # type_ = graph_tensor[:, -1]
            # denoisedKG = (index_.t().long().cuda(), type_.long().cuda())
            # self.model.setDenoisedKG(denoisedKG)

        with torch.no_grad():
            index_, type_ = denoisedKG
            mask = ((torch.rand(type_.shape[0]) + self.model.keepRate).floor()).type(torch.bool)
            denoisedKG = (index_[:, mask], type_[mask])
            self.generatedKG = denoisedKG
        
        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        scaler = amp.GradScaler(enabled=self.enable_scaler)
        for batch_idx, interaction in enumerate(iter_data):
            # interaction = list(map(lambda x: x.to(self.device), tem))
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction,denoisedKG)
            # losses.backward()
            # self.optimizer.step()
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
            self._check_nan(loss)
            scaler.scale(loss + sync_loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(
            valid_data, load_best_model=False, show_progress=show_progress
        )
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        if not self.config["single_spec"] and self.config["local_rank"] != 0:
            return
        saved_model_file = kwargs.pop("saved_model_file", self.saved_model_file)
        state = {
            "config": self.config,
            "epoch": epoch,
            "cur_step": self.cur_step,
            "best_valid_score": self.best_valid_score,
            "state_dict": self.model.state_dict(),
            "other_parameter": self.model.other_parameter(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file, pickle_protocol=4)
        if verbose:
            self.logger.info(
                set_color("Saving current", "blue") + f": {saved_model_file}"
            )

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        self.saved_model_file = resume_file
        checkpoint = torch.load(resume_file, map_location=self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.cur_step = checkpoint["cur_step"]
        self.best_valid_score = checkpoint["best_valid_score"]

        # load architecture params from checkpoint
        if checkpoint["config"]["model"].lower() != self.config["model"].lower():
            self.logger.warning(
                "Architecture configuration given in config file is different from that of checkpoint. "
                "This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.load_other_parameter(checkpoint.get("other_parameter"))

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        message_output = "Checkpoint loaded. Resume training from epoch {}".format(
            self.start_epoch
        )
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config["loss_decimal_place"] or 4
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = set_color("train_loss%d", "blue") + ": %." + str(des) + "f"
            train_loss_output += ", ".join(
                des % (idx + 1, loss) for idx, loss in enumerate(losses)
            )
        else:
            des = "%." + str(des) + "f"
            train_loss_output += set_color("train loss", "blue") + ": " + des % losses
        return train_loss_output + "]"

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag="Loss/Train"):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            "learner": self.config["learner"],
            "learning_rate": self.config["learning_rate"],
            "train_batch_size": self.config["train_batch_size"],
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter
            for parameters in self.config.parameters.values()
            for parameter in parameters
        }.union({"model", "dataset", "config_files", "device"})
        # other model-specific hparam
        hparam_dict.update(
            {
                para: val
                for para, val in self.config.final_config_dict.items()
                if para not in unrecorded_parameter
            }
        )
        for k in hparam_dict:
            if hparam_dict[k] is not None and not isinstance(
                hparam_dict[k], (bool, str, float, int)
            ):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(
            hparam_dict, {"hparam/best_valid_result": best_valid_result}
        )

    def fit(
        self,
        train_data,
        # train_dataloader,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )
                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def _full_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data
        try:
            # Note: interaction without item ids
            scores = self.model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i

    def _neg_sample_batch_eval(self, batched_data):
        interaction, row_idx, positive_u, positive_i = batched_data
        batch_size = interaction.length
        if batch_size <= self.test_batch_size:
            origin_scores = self.model.predict(interaction.to(self.device))
        else:
            origin_scores = self._spilt_predict(interaction, batch_size)

        if self.config["eval_type"] == EvaluatorType.VALUE:
            return interaction, origin_scores, positive_u, positive_i
        elif self.config["eval_type"] == EvaluatorType.RANKING:
            col_idx = interaction[self.config["ITEM_ID_FIELD"]]
            batch_user_num = positive_u[-1] + 1
            scores = torch.full(
                (batch_user_num, self.tot_item_num), -np.inf, device=self.device
            )
            scores[row_idx, col_idx] = origin_scores
            return interaction, scores, positive_u, positive_i

    @torch.no_grad()
    def evaluate(
        self, eval_data, load_best_model=False, model_file=None, show_progress=False
    ):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)

        self.model.eval()

        #if isinstance(eval_data, FullSortEvalDataLoader): 只能全排列ranking测试了
        if 1 == 1 :
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data._dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result

    def _map_reduce(self, result, num_sample):
        gather_result = {}
        total_sample = [
            torch.zeros(1).to(self.device) for _ in range(self.config["world_size"])
        ]
        torch.distributed.all_gather(
            total_sample, torch.Tensor([num_sample]).to(self.device)
        )
        total_sample = torch.cat(total_sample, 0)
        total_sample = torch.sum(total_sample).item()
        for key, value in result.items():
            result[key] = torch.Tensor([value * num_sample]).to(self.device)
            gather_result[key] = [
                torch.zeros_like(result[key]).to(self.device)
                for _ in range(self.config["world_size"])
            ]
            torch.distributed.all_gather(gather_result[key], result[key])
            gather_result[key] = torch.cat(gather_result[key], dim=0)
            gather_result[key] = round(
                torch.sum(gather_result[key]).item() / total_sample,
                self.config["metric_decimal_place"],
            )
        return gather_result

    def _spilt_predict(self, interaction, batch_size):
        spilt_interaction = dict()
        for key, tensor in interaction.interaction.items():
            spilt_interaction[key] = tensor.split(self.test_batch_size, dim=0)
        num_block = (batch_size + self.test_batch_size - 1) // self.test_batch_size
        result_list = []
        for i in range(num_block):
            current_interaction = dict()
            for key, spilt_tensor in spilt_interaction.items():
                current_interaction[key] = spilt_tensor[i]
            result = self.model.predict(
                Interaction(current_interaction).to(self.device)
            )
            if len(result.shape) == 0:
                result = result.unsqueeze(0)
            result_list.append(result)
        return torch.cat(result_list, dim=0)
    
    
    
class diffkgKGTrainer(Trainer):
    r"""KGTrainer is designed for Knowledge-aware recommendation methods. Some of these models need to train the
    recommendation related task and knowledge related task alternately.

    """

    def __init__(self, config, model):
        super(diffkgKGTrainer, self).__init__(config, model)

        self.train_rec_step = config["train_rec_step"]
        self.train_kg_step = config["train_kg_step"]
        

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if self.train_rec_step is None or self.train_kg_step is None:
            interaction_state = KGDataLoaderState.RSKG
        elif (
            epoch_idx % (self.train_rec_step + self.train_kg_step) < self.train_rec_step
        ):
            interaction_state = KGDataLoaderState.RS
        else:
            interaction_state = KGDataLoaderState.KG
        if not self.config["single_spec"]:
            train_data.knowledge_shuffle(epoch_idx)
        train_data.set_mode(interaction_state)
        if interaction_state in [KGDataLoaderState.RSKG, KGDataLoaderState.RS]:
            return super()._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
        elif interaction_state in [KGDataLoaderState.KG]:
            return super()._train_epoch(
                train_data,
                epoch_idx,
                loss_func=self.model.calculate_kg_loss,
                show_progress=show_progress,
            )
        return None
    
    
import torch.utils.data as data

class DiffusionData(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        return torch.FloatTensor(item), index
    
    def __len__(self):
        return len(self.data)
