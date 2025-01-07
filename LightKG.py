import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.init import xavier_uniform_initialization
from recbole.model.abstract_recommender import KnowledgeRecommender, GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from recbole.model.layers import SparseDropout
import math
import torch_sparse
class LightKG(KnowledgeRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightKG, self).__init__(config, dataset)
        self.embedding_size = config["embedding_size"]
        self.layer = config["layer"]
        self.mess_dropout_rate = config["mess_dropout_rate"]
        self.cos_loss = config["cos_loss"] #Control whether to use the contrastive layer.
        self.item_loss = config["item_loss"]
        self.user_loss = config["user_loss"]
        self.temperature = 0.6
        self.mess_dropout = nn.Dropout(p=self.mess_dropout_rate)  
        self.inter = dataset.inter_matrix(form="coo").astype(
            np.float32
        )
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities,self.embedding_size)
        self.relation_embedding_trane = nn.Embedding(self.n_relations+1, self.embedding_size)
        self.kg_graph = dataset.kg_graph(
            form="coo", value_field="relation_id"
        )
        self.relation_embedding = nn.Embedding(2*self.n_relations-1,1)
        self.CKG = self.get_ckg() 
        self.du = self.get_D() # Matrix D records the number of neighbors for each node.
        self.Sim = self.get_Sim() # Matrix Sim records the similarities between users and users, and items and items.
        self.apply(xavier_uniform_initialization)
        self.mf_loss = BPRLoss()
        self.test = False
        self.all_user_embeddings,self.all_entity_embeddings = self.user_embedding.weight.to(self.device),self.entity_embedding.weight.to(self.device)
    
    def get_Sim(self):

        indice = self.CKG.coalesce().indices()
        indice = indice[:,indice[0]<self.n_users+self.n_items+1]
        indice = indice[:,indice[1]<self.n_users+self.n_items+1]
        values = torch.tensor([1]*indice.shape[1]).to(self.device)
        b = torch.sparse_coo_tensor(indice,values).to(torch.float16)

        c = torch.sparse.mm(b,b)
        c = c + b
        c = c.coalesce()
        indices = c.indices()
        values = c.values()
        
        D = self.du[indices][0] * self.du[indices][1]
        value = -values * D
        
        B = torch.sparse_coo_tensor(indices,value,size=[self.n_users+self.n_items+1,self.n_users+self.n_items+1])
        return B
        
    def get_ckg(self):
        kg_hail = self.kg_graph.row
        kg_tail = self.kg_graph.col
        kg_relation = self.kg_graph.data
        kg_relation_plus = kg_relation + self.n_relations-2
        kg_hail_plus = np.concatenate((kg_hail, kg_tail))
        kg_tail_plus = np.concatenate((kg_tail, kg_hail))
        kg_relation_plus = np.concatenate((kg_relation, kg_relation_plus))
        
        inter_hail = self.inter.row
        inter_tail = self.inter.col
        inter_relation = np.array([(self.n_relations-2) * 2 +1] * len(inter_hail) )
        inter_relation_plus = np.array([(self.n_relations-2) * 2 +2] * len(inter_hail) )
        
        hail = torch.tensor(np.concatenate((inter_hail,inter_tail+self.n_users,kg_hail_plus+self.n_users)))
        tail = torch.tensor(np.concatenate((inter_tail+self.n_users,inter_hail,kg_tail_plus+self.n_users)))
        relation = torch.tensor(np.concatenate((inter_relation,inter_relation_plus,kg_relation_plus)))
        size = torch.Size([self.n_entities+self.n_users, self.n_entities+self.n_users])
        A = torch.sparse_coo_tensor(torch.stack([hail,tail]),relation,size)
        return A.to(self.device)
    
    def get_D(self): #get degree of each node
        inter_user = torch.tensor(self.inter.row)
        inter_item = torch.tensor(self.inter.col)
        data = torch.tensor([1]*len(inter_item))
        inter = torch.sparse_coo_tensor(torch.stack([inter_user,inter_item]),data,size=torch.Size([self.n_users,self.n_items]))
        user_degree = torch.sparse.sum(inter,1).to_dense() 
        item_degree_1 = torch.sparse.sum(inter,0).to_dense() 
        
        kg_hail = self.kg_graph.row
        kg_tail = self.kg_graph.col
        kg_hail_plus = np.concatenate((kg_hail, kg_tail))
        kg_tail_plus = np.concatenate((kg_tail, kg_hail))
        e = list(zip(kg_hail_plus,kg_tail_plus))
        e = list(set(e))
        head = torch.tensor( [x[0] for x in e] )
        tail = torch.tensor( [x[1] for x in e] )
        data = torch.tensor([1] *len(head))
        kg  = torch.sparse_coo_tensor(torch.stack([head,tail]),data)
        entity_degree = torch.sparse.sum(kg,1).to_dense() 
        item_degree_2 = entity_degree[0:self.n_items]
        item_degree = item_degree_1 + item_degree_2 
        du = torch.cat((user_degree,item_degree))
        du = torch.cat((du,entity_degree[self.n_items:]))
        du = 1 / torch.sqrt(du)
        du = torch.where(torch.isinf(du), torch.tensor(0.0), du)   
        return du.to(self.device)

        
    def get_normal_matrix(self):
        #A = self.node_dropout(self.CKG)
        data = self.CKG._values()
        value_A = self.relation_embedding(data).view(-1)
        indices_A = self.CKG._indices()
        D = self.du[indices_A][0] * self.du[indices_A][1]
        value_A = value_A * D
        #A = torch.sparse_coo_tensor(indices_A,value_A).coalesce()
        #A = self.node_dropout(A)
        return indices_A,value_A
        
        
    def get_ego_embeddings(self):  
        user_emb = self.user_embedding.weight
        entity_emb = self.entity_embedding.weight
        if self.mess_dropout_rate > 0.0 and self.test == False:
                entity_emb = self.mess_dropout(entity_emb)
                user_emb = self.mess_dropout(user_emb)
        ego_embeddings = torch.cat([user_emb, entity_emb], dim=0)
        return ego_embeddings
    
    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        myself = all_embeddings.clone()
        embeddings_list = [all_embeddings]
        normal_matrix_index, normal_matrix_value = self.get_normal_matrix()
        for layer_idx in range(self.layer):
            all_embeddings = torch_sparse.spmm(normal_matrix_index,normal_matrix_value,self.n_entities + self.n_users,self.n_entities + self.n_users, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_lightgcn_embeddings, entity_lightgcn_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_entities]
        )
        return user_lightgcn_embeddings, entity_lightgcn_embeddings
    
    def _get_rec_embedding(self, user, pos_item, neg_item):
        user_e = self.user_embedding(user)
        pos_item_e = self.entity_embedding(pos_item)
        neg_item_e = self.entity_embedding(neg_item)
        rec_r_e = self.relation_embedding_trane.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        return user_e, pos_item_e, neg_item_e, rec_r_e

    def _get_kg_embedding(self, head, pos_tail, neg_tail, relation):
        head_e = self.entity_embedding(head)
        pos_tail_e = self.entity_embedding(pos_tail)
        neg_tail_e = self.entity_embedding(neg_tail)
        relation_e = self.relation_embedding_trane(relation)
        return head_e, pos_tail_e, neg_tail_e, relation_e
    
    def get_user_xishu(self,jiedian):
        t = self.Sim.index_select(0,index=jiedian)
        t = t.transpose(0,1)
        t = t.index_select(0,index=jiedian).to_dense()
        #t = torch.sigmoid(t)*2
        t = 1 + t
        return t
    
    def get_user_loss(self,jiedian,embedding):
        xishu = self.get_user_xishu(jiedian)
        normal_embedding = F.normalize(embedding,p=2,dim=1)
        du = self.du[jiedian]
        matrix = du.view(-1,1) @ du.view(1,-1)
        matrix = 1 - matrix
        loss = torch.sum(matrix * torch.exp((normal_embedding @ normal_embedding.T)*xishu/self.temperature)) 
        return loss
    
    def get_item_xishu(self,pos_item,neg_item):
        t = self.Sim.index_select(0,index=pos_item)
        t = t.transpose(0,1)
        t = t.index_select(0,index=neg_item).to_dense()
        t = t.transpose(0,1)
        t = 1 + t
        return t
    
    def get_item_loss(self,pos_item,pos_e,neg_item,neg_e):
        xishu = self.get_item_xishu(pos_item,neg_item)
        normal_pos_embedding = F.normalize(pos_e,p=2,dim=1)
        normal_neg_embedding = F.normalize(neg_e,p=2,dim=1)
        pos_du = self.du[pos_item]
        neg_du = self.du[neg_item]
        matrix = pos_du.view(-1,1) @ neg_du.view(1,-1)
        matrix = 1 - matrix
        loss = torch.sum(matrix * torch.exp((normal_pos_embedding @ normal_neg_embedding.T)*xishu/self.temperature))
        return loss
    
    def calculate_loss(self, interaction):
        self.test = False
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        user_lightgcn_embeddings, item_lightgcn_embeddings = self.forward()
        self.all_user_embeddings,self.all_entity_embeddings = user_lightgcn_embeddings, item_lightgcn_embeddings
        u_embeddings = self.all_user_embeddings[user]
        pos_embeddings = self.all_entity_embeddings[pos_item]
        neg_embeddings = self.all_entity_embeddings[neg_item]
        pos_score = torch.mul(u_embeddings,pos_embeddings).sum(dim=1)
        neg_score = torch.mul(u_embeddings,neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_score,neg_score)
        

        user_e, pos_item_e, neg_item_e, rec_r_e = self._get_rec_embedding(
            user, pos_item, neg_item
        )
        cos_loss = 0
        if self.cos_loss !=0:
            user_loss = self.get_user_loss(user,user_e)
            item_loss = self.get_item_loss(pos_item+self.n_users,pos_item_e,neg_item+self.n_users,neg_item_e)
            cos_loss = self.user_loss * user_loss + self.item_loss * item_loss 
        loss = mf_loss  + cos_loss
        return loss 
    
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if self.test == False:
            self.test = True
            self.all_user_embeddings,self.all_entity_embeddings = self.forward()

        user_lightgcn_embeddings, item_lightgcn_embeddings = self.all_user_embeddings,self.all_entity_embeddings
        
        scores = torch.mul(user_lightgcn_embeddings[user],item_lightgcn_embeddings[item]).sum(dim=1)
        return scores
        
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.test == False:
            self.test = True
            self.all_user_embeddings,self.all_entity_embeddings = self.forward()
        u_embeddings = self.all_user_embeddings[user]
        i_embeddings = self.all_entity_embeddings[: self.n_items]
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)
        