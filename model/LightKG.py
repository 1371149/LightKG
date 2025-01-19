# Import necessary libraries
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
        """
        Initialize the LightKG model.
        Args:
            config: Configuration object containing model hyperparameters.
            dataset: Dataset object containing user-item and KG data.
        """
        super(LightKG, self).__init__(config, dataset)
        # Model hyperparameters
        self.embedding_size = config["embedding_size"]
        self.layer = config["layer"]  # Number of GCN layers
        self.mess_dropout_rate = config["mess_dropout_rate"]  # Dropout rate for message passing
        self.cos_loss = config["cos_loss"]  # Whether to use contrastive loss
        self.beta_i = config["item_loss"]  # Weight for item contrastive loss
        self.beta_u = config["user_loss"]  # Weight for user contrastive loss
        self.temperature = 0.6  # Temperature parameter for contrastive loss
        
        # Dropout layer
        self.mess_dropout = nn.Dropout(p=self.mess_dropout_rate)
        
        # Interaction matrix (user-item interactions)
        self.inter = dataset.inter_matrix(form="coo").astype(np.float32)

        # Embedding layers
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(2 * self.n_relations - 1, 1)
        
        # Knowledge graph data
        self.kg_graph = dataset.kg_graph(form="coo", value_field="relation_id")
        
        # Construct CKG (Collaborative Knowledge Graph)
        self.CKG = self.get_ckg() 
        
        # Degree matrix for normalization
        self.Degree = self.get_degree_matrix()
        
        # Similarity matrix for contrastive loss
        self.Similarity_matrix = self.get_Similarity_matrix()
        
        # Initialize weights
        self.apply(xavier_uniform_initialization)
        
        # Loss functions
        self.mf_loss = BPRLoss()  # Bayesian Personalized Ranking loss
        self.test = False  # Flag for testing mode

        # Store embeddings for testing
        self.all_user_embeddings, self.all_entity_embeddings = self.user_embedding.weight.to(self.device), self.entity_embedding.weight.to(self.device)
    
    def get_Similarity_matrix(self):
        """
        Compute the similarity matrix for contrastive learning.
        The similarity matrix captures relationships between users and items in the CKG (Collaborative Knowledge Graph).
        
        Returns:
            A sparse tensor representing the similarity matrix.
        """
        # Step 1: Extract indices from the CKG (Collaborative Knowledge Graph)
        ckg_indices = self.CKG.coalesce().indices()

        # Step 2: Filter indices to include only users and items (ignore entities)
        user_item_mask = (ckg_indices[0] < self.n_users + self.n_items + 1) & (ckg_indices[1] < self.n_users + self.n_items + 1)
        filtered_indices = ckg_indices[:, user_item_mask]

        # Step 3: Create a sparse tensor for user-item interactions
        interaction_values = torch.ones(filtered_indices.shape[1], device=self.device)
        interaction_matrix = torch.sparse_coo_tensor(filtered_indices, interaction_values, dtype=torch.float16)

        # Step 4: Compute the similarity matrix by multiplying the interaction matrix with itself
        similarity_matrix = torch.sparse.mm(interaction_matrix, interaction_matrix)

        # Step 5: Add the original interaction matrix to capture direct relationships
        similarity_matrix = similarity_matrix + interaction_matrix
        similarity_matrix = similarity_matrix.coalesce()  # Remove duplicates

        # Step 6: Extract indices and values from the similarity matrix
        similarity_indices = similarity_matrix.indices()
        similarity_values = similarity_matrix.values()

        # Step 7: Normalize the similarity values using node degrees
        degree_product = self.Degree[similarity_indices][0] * self.Degree[similarity_indices][1]
        normalized_values = -similarity_values * degree_product

        # Step 8: Construct the final similarity matrix
        final_similarity_matrix = torch.sparse_coo_tensor(
            similarity_indices,
            normalized_values,
            size=[self.n_users + self.n_items + 1, self.n_users + self.n_items + 1]
        )

        return final_similarity_matrix
        
    def get_ckg(self):
        """
        Construct the Collaborative Knowledge Graph (CKG) by combining user-item interactions and the knowledge graph (KG).
        The CKG is a sparse tensor that represents relationships between users, items, and entities in the KG.

        Returns:
            torch.sparse.Tensor: A sparse tensor representing the CKG.
        """
        # Extract KG data: head entities, tail entities, and relations
        kg_head = self.kg_graph.row  # Head entities in the KG
        kg_tail = self.kg_graph.col  # Tail entities in the KG
        kg_relation = self.kg_graph.data  # Relations in the KG

        # Create reverse relations for bidirectional edges in the KG
        kg_relation_reversed = kg_relation + (self.n_relations - 2)  # Reverse relation IDs

        # Combine original and reversed KG edges
        kg_head_bidirectional = np.concatenate((kg_head, kg_tail))  # Head entities for both directions
        kg_tail_bidirectional = np.concatenate((kg_tail, kg_head))  # Tail entities for both directions
        kg_relation_bidirectional = np.concatenate((kg_relation, kg_relation_reversed))  # Relations for both directions

        # Extract user-item interaction data
        inter_head = self.inter.row  # Users in interactions
        inter_tail = self.inter.col  # Items in interactions

        # Define relation IDs for user-item interactions
        inter_relation = np.array([(self.n_relations - 2) * 2 + 1] * len(inter_head))  # Relation ID for user->item
        inter_relation_reversed = np.array([(self.n_relations - 2) * 2 + 2] * len(inter_head))  # Relation ID for item->user

        # Combine all head entities (users, items, and KG entities)
        all_head = torch.tensor(np.concatenate((
            inter_head,  # Users
            inter_tail + self.n_users,  # Items (shifted by number of users)
            kg_head_bidirectional + self.n_users  # KG entities (shifted by number of users)
        )))

        # Combine all tail entities (items, users, and KG entities)
        all_tail = torch.tensor(np.concatenate((
            inter_tail + self.n_users,  # Items (shifted by number of users)
            inter_head,  # Users
            kg_tail_bidirectional + self.n_users  # KG entities (shifted by number of users)
        )))

        # Combine all relations (user-item, item-user, and KG relations)
        all_relation = torch.tensor(np.concatenate((
            inter_relation,  # User->item relations
            inter_relation_reversed,  # Item->user relations
            kg_relation_bidirectional  # KG relations
        )))

        # Define the size of the CKG sparse tensor
        ckg_size = torch.Size([self.n_entities + self.n_users, self.n_entities + self.n_users])

        # Create the CKG sparse tensor
        ckg_sparse_tensor = torch.sparse_coo_tensor(
            torch.stack([all_head, all_tail]),  # Indices for head and tail entities
            all_relation,  # Relation values
            size=ckg_size  # Size of the tensor
        )

        # Move the tensor to the appropriate device (e.g., GPU)
        return ckg_sparse_tensor.to(self.device)
    
    def get_degree_matrix(self):
        """
        Compute the degree matrix for normalization.
        The degree matrix records the number of neighbors for each node in the graph.
        Returns:
            A tensor containing the degree of each node, normalized by 1/sqrt(degree).
        """
        # Step 1: Compute degrees for user-item interactions
        # Convert interaction rows and columns to tensors
        inter_user = torch.tensor(self.inter.row)  # User indices in interactions
        inter_item = torch.tensor(self.inter.col)  # Item indices in interactions
        data = torch.tensor([1] * len(inter_item))  # All interactions have a value of 1

        # Create a sparse tensor for user-item interactions
        inter_sparse = torch.sparse_coo_tensor(
            torch.stack([inter_user, inter_item]),  # Indices for sparse tensor
            data,  # Values for sparse tensor
            size=torch.Size([self.n_users, self.n_items])  # Size of the sparse tensor
        )

        # Compute user degrees (sum of interactions per user)
        user_degree = torch.sparse.sum(inter_sparse, dim=1).to_dense()  # Sum over rows
        # Compute item degrees (sum of interactions per item)
        item_degree_inter = torch.sparse.sum(inter_sparse, dim=0).to_dense()  # Sum over columns

        # Step 2: Compute degrees for knowledge graph (KG) entities
        kg_head = self.kg_graph.row  # Head entities in KG
        kg_tail = self.kg_graph.col  # Tail entities in KG

        # Create bidirectional edges for KG
        kg_head_bidirectional = np.concatenate((kg_head, kg_tail))  # Combine head and tail
        kg_tail_bidirectional = np.concatenate((kg_tail, kg_head))  # Combine tail and head

        # Remove duplicate edges
        edges = list(zip(kg_head_bidirectional, kg_tail_bidirectional))  # Pair head and tail
        edges = list(set(edges))  # Remove duplicates

        # Convert edges to tensors
        head_indices = torch.tensor([x[0] for x in edges])  # Head indices of edges
        tail_indices = torch.tensor([x[1] for x in edges])  # Tail indices of edges
        edge_data = torch.tensor([1] * len(head_indices))  # All edges have a value of 1

        # Create a sparse tensor for KG edges
        kg_sparse = torch.sparse_coo_tensor(
            torch.stack([head_indices, tail_indices]),  # Indices for sparse tensor
            edge_data  # Values for sparse tensor
        )

        # Compute entity degrees (sum of edges per entity)
        entity_degree = torch.sparse.sum(kg_sparse, dim=1).to_dense()  # Sum over rows

        # Step 3: Combine degrees for users, items, and entities
        # Item degrees from KG (only for items present in the KG)
        item_degree_kg = entity_degree[:self.n_items]
        # Total item degrees (sum of interaction degrees and KG degrees)
        item_degree_total = item_degree_inter + item_degree_kg

        # Combine user degrees, item degrees, and remaining entity degrees
        degree_matrix = torch.cat((user_degree, item_degree_total))  # Users and items
        degree_matrix = torch.cat((degree_matrix, entity_degree[self.n_items:]))  # Remaining entities

        # Step 4: Normalize degrees
        degree_matrix = 1 / torch.sqrt(degree_matrix)  # Normalize by 1/sqrt(degree)
        # Handle infinite values (for nodes with zero degree)
        degree_matrix = torch.where(torch.isinf(degree_matrix), torch.tensor(0.0), degree_matrix)

        # Return the degree matrix, moved to the appropriate device (e.g., GPU)
        return degree_matrix.to(self.device)

        
    def get_normal_matrix(self):
        """
        Compute the normalized adjacency matrix for message passing in the graph.
        The normalization is done using the degree matrix to scale the relation embeddings.

        Returns:
            indices_A (torch.Tensor): Indices of the normalized adjacency matrix.
            value_A (torch.Tensor): Values of the normalized adjacency matrix.
        """
        # Extract the values (relation embeddings) from the Collaborative Knowledge Graph (CKG)
        relation_values = self.CKG._values()

        # Retrieve the relation embeddings and flatten them into a 1D tensor
        relation_embeddings = self.relation_embedding(relation_values).view(-1)

        # Get the indices of the edges in the CKG
        edge_indices = self.CKG._indices()

        # Compute the degree normalization factor for each edge
        # Degree normalization is applied to both nodes connected by the edge
        degree_normalization = self.Degree[edge_indices][0] * self.Degree[edge_indices][1]

        # Apply degree normalization to the relation embeddings
        normalized_values = relation_embeddings * degree_normalization

        return edge_indices, normalized_values
        
        
    def get_all_embeddings(self):
        """
        Retrieve and concatenate the embeddings for users and entities.
        Applies dropout to embeddings during training if dropout rate is greater than 0.
        
        Returns:
            torch.Tensor: Concatenated embeddings of users and entities.
        """
        # Retrieve user and entity embeddings from the embedding layers
        user_embeddings = self.user_embedding.weight  # Embeddings for users
        entity_embeddings = self.entity_embedding.weight  # Embeddings for entities

        # Apply dropout to embeddings during training (if dropout rate > 0 and not in test mode)
        if self.mess_dropout_rate > 0.0 and not self.test:
            entity_embeddings = self.mess_dropout(entity_embeddings)  # Apply dropout to entity embeddings
            user_embeddings = self.mess_dropout(user_embeddings)  # Apply dropout to user embeddings

        # Concatenate user and entity embeddings along the first dimension
        all_embeddings = torch.cat([user_embeddings, entity_embeddings], dim=0)

        return all_embeddings
    
    def forward(self):
        """
        Perform the forward pass of the LightKG model.
        This method propagates user and entity embeddings through multiple GNN layers
        and aggregates the results to produce final embeddings.

        Returns:
            user_embeddings: Embeddings for users after GNN.
            entity_embeddings: Embeddings for entities after GNN.
        """
        # Step 1: Get initial all embeddings (user and entity embeddings)
        initial_embeddings = self.get_all_embeddings()
        
        # Step 2: Store embeddings for each layer
        layer_embeddings = [initial_embeddings]  # List to store embeddings from each GCN layer
        
        # Step 3: Get the normalized adjacency matrix for message passing
        adjacency_indices, adjacency_values = self.get_normal_matrix()
        
        # Step 4: Perform message passing through each GCN layer
        for layer_idx in range(self.layer):
            # Propagate embeddings using sparse matrix multiplication
            initial_embeddings = torch_sparse.spmm(
                adjacency_indices, 
                adjacency_values, 
                self.n_entities + self.n_users, 
                self.n_entities + self.n_users, 
                initial_embeddings
            )
            # Store the embeddings for the current layer
            layer_embeddings.append(initial_embeddings)
        
        # Step 5: Aggregate embeddings from all layers
        # Stack embeddings from all layers and compute their mean
        aggregated_embeddings = torch.stack(layer_embeddings, dim=1)
        aggregated_embeddings = torch.mean(aggregated_embeddings, dim=1)
        
        # Step 6: Split aggregated embeddings into user and entity embeddings
        user_embeddings, entity_embeddings = torch.split(
            aggregated_embeddings, 
            [self.n_users, self.n_entities]
        )
        
        return user_embeddings, entity_embeddings
    
    def _get_rec_embedding(self, user, pos_item, neg_item):
        """
        Retrieve 0-layer embeddings for users, positive items, and negative items.
        """
        user_e = self.user_embedding(user)
        pos_item_e = self.entity_embedding(pos_item)
        neg_item_e = self.entity_embedding(neg_item)
        return user_e, pos_item_e, neg_item_e
    
    def get_user_similarity(self,node):
        """
        Compute the similarity matrix for a given set of user nodes.
        Args:
            node (torch.Tensor): Indices of the user nodes.
        Returns:
            torch.Tensor: A dense similarity matrix for the given user nodes.
        """
        sim = self.Similarity_matrix.index_select(0,index=node)
        sim = sim.transpose(0,1)
        sim = sim.index_select(0,index=node).to_dense()
        #t = torch.sigmoid(t)*2
        sim = 1 + sim
        return sim
    
    def get_user_loss(self,node,embedding):
        """
        Compute the contrastive loss for users based on their embeddings and similarities.
        Args:
            node (torch.Tensor): Indices of the user nodes.
            embedding (torch.Tensor): Embeddings of the user nodes.
        Returns:
            torch.Tensor: The computed user contrastive loss.
        """
        sim = self.get_user_similarity(node)
        normal_embedding = F.normalize(embedding,p=2,dim=1)
        degree = self.Degree[node]
        matrix = degree.view(-1,1) @ degree.view(1,-1)
        matrix = 1 - matrix
        loss = torch.sum(matrix * torch.exp((normal_embedding @ normal_embedding.T)*sim/self.temperature)) 
        return loss
    
    def get_item_similarity(self,pos_item,neg_item):
        """
        Compute the similarity matrix between positive and negative items.
        Args:
            pos_item (torch.Tensor): Indices of the positive items.
            neg_item (torch.Tensor): Indices of the negative items.
        Returns:
            torch.Tensor: A dense similarity matrix between positive and negative items.
        """
        sim = self.Similarity_matrix.index_select(0,index=pos_item)
        sim = sim.transpose(0,1)
        sim = sim.index_select(0,index=neg_item).to_dense()
        sim = sim.transpose(0,1)
        sim = 1 + sim
        return sim
    
    def get_item_loss(self,pos_item,pos_e,neg_item,neg_e):
        """
        Compute the contrastive loss for items based on their embeddings and similarities.
        Args:
            pos_item (torch.Tensor): Indices of the positive items.
            pos_embedding (torch.Tensor): Embeddings of the positive items.
            neg_item (torch.Tensor): Indices of the negative items.
            neg_embedding (torch.Tensor): Embeddings of the negative items.
        Returns:
            torch.Tensor: The computed item contrastive loss.
        """
        sim = self.get_item_similarity(pos_item,neg_item)
        normal_pos_embedding = F.normalize(pos_e,p=2,dim=1)
        normal_neg_embedding = F.normalize(neg_e,p=2,dim=1)
        pos_degree = self.Degree[pos_item]
        neg_degree = self.Degree[neg_item]
        matrix = pos_degree.view(-1,1) @ neg_degree.view(1,-1)
        matrix = 1 - matrix
        loss = torch.sum(matrix * torch.exp((normal_pos_embedding @ normal_neg_embedding.T)*sim/self.temperature))
        return loss
    
    def calculate_loss(self, interaction):
        """
        Calculate the total loss for training, including BPR loss and contrastive loss (if enabled).
        
        Args:
            interaction (dict): Contains user, positive item, and negative item IDs.
            
        Returns:
            torch.Tensor: The total loss value.
        """
        # Set test mode to False (training mode)
        self.test = False

        # Extract user, positive item, and negative item IDs from interaction
        user_ids = interaction[self.USER_ID]
        pos_item_ids = interaction[self.ITEM_ID]
        neg_item_ids = interaction[self.NEG_ITEM_ID]
        
         # Forward pass to get user and item embeddings
        user_embeddings, item_embeddings = self.forward()
        self.all_user_embeddings, self.all_entity_embeddings = user_embeddings, item_embeddings

        # Get embeddings for users, positive items, and negative items
        user_emb = self.all_user_embeddings[user_ids]
        pos_item_emb = self.all_entity_embeddings[pos_item_ids]
        neg_item_emb = self.all_entity_embeddings[neg_item_ids]

        # Compute positive and negative scores for BPR loss
        pos_scores = torch.mul(user_emb, pos_item_emb).sum(dim=1)  # Dot product of user and positive item embeddings
        neg_scores = torch.mul(user_emb, neg_item_emb).sum(dim=1)  # Dot product of user and negative item embeddings

        # Compute Bayesian Personalized Ranking (BPR) loss
        bpr_loss = self.mf_loss(pos_scores, neg_scores)

        # Get embeddings for contrastive loss (if enabled)
        user_emb_rec, pos_item_emb_rec, neg_item_emb_rec = self._get_rec_embedding(user_ids, pos_item_ids, neg_item_ids)

        # Initialize contrastive loss to 0
        contrastive_loss = 0

        # Compute contrastive loss if enabled
        if self.cos_loss != 0:
            # Compute user contrastive loss
            user_contrastive_loss = self.get_user_loss(user_ids, user_emb_rec)

            # Compute item contrastive loss
            item_contrastive_loss = self.get_item_loss(
                pos_item_ids + self.n_users, pos_item_emb_rec,
                neg_item_ids + self.n_users, neg_item_emb_rec
            )

            # Weighted sum of user and item contrastive losses
            contrastive_loss = self.beta_u * user_contrastive_loss + self.beta_i * item_contrastive_loss

        # Total loss = BPR loss + contrastive loss
        total_loss = bpr_loss + contrastive_loss

        return total_loss
    
    def predict(self, interaction):
        """
        Predict the interaction score between a user and an item.
        
        Args:
            interaction (dict): A dictionary containing user and item IDs.
            
        Returns:
            torch.Tensor: Predicted scores for the user-item pairs.
        """
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
        
