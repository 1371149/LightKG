# Import necessary libraries
import random
from random import randint
import torch
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import copy
import torch.nn as nn
from recbole.utils import init_logger, init_seed
from recbole.data.dataset.kg_dataset import KnowledgeBasedDataset
from recbole.data.dataloader.knowledge_dataloader import KnowledgeBasedDataLoader
from recbole.config.configurator import Config
from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from logging import getLogger
from recbole.trainer import KGTrainer, Trainer
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
import torch.profiler as profiler
import torch.nn.functional as F
from recbole.quick_start.quick_start import load_data_and_model

# Import custom models
from LightKG import *
from CFKG import *
from KGAT import *
from KGRec import *
from CKE import *
from RippleNet import *
from KGIN import *
from KGCN import *
from LightGCN import *
from MCCLK import *

# Configuration setup
config_dict = {'seed': 2020}  # Set initial seed for reproducibility
config_file_list = ['./yaml/ml-1m_LightKG.yaml']  # Load configuration file. If you want to use other configuration, you should change "config_file_list".

# Initialize configuration
config = Config(model=LightKG, dataset='ml-1m', config_file_list=config_file_list, config_dict=config_dict) #If you want to user other dataset, change "Config( dataset='target dataset')" 

# Set seed for reproducibility
init_seed(config['seed'], config['reproducibility'])

# Initialize logger
init_logger(config)
logger = getLogger()

# Create dataset and prepare data
data = create_dataset(config)
logger.info(data)  # Log dataset information
train_data, valid_data, test_data = data_preparation(config, data)


# Initialize the model
model = LightKG(config=config, dataset=train_data._dataset).to(config['device'])

# Log model information
logger.info(model)

# Initialize trainer
trainer = KGTrainer(config, model)

# Train the model and evaluate on validation set
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

# Evaluate the model on the test set
test_result = trainer.evaluate(test_data)

# Log the best validation and test results
logger.info('Best valid result: {}'.format(best_valid_result))
logger.info('Test result: {}'.format(test_result))
