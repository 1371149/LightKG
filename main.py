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
import argparse

# Import custom models
from model.LightKG import *
from model.CFKG import *
from model.KGAT import *
from model.CKE import *
from model.RippleNet import *
from model.KGIN import *
from model.KGCN import *
from model.LightGCN import *
from model.MCCLK import *



parser = argparse.ArgumentParser(description="Run LightKG")
parser.add_argument("--dataset", type=str,default="lastfm", help="choose dataset from ml-1m, lastfm, Amazon-book, book-crossing.")
args = parser.parse_args()

# Configuration setup
config_dict = {'seed': 2020}  # Set initial seed for reproducibility

# Load configuration file
if args.dataset == "lastfm":
    config_file_list = ['./yaml/lastfm_lightkg.yaml']
elif args.dataset == "ml-1m":
    config_file_list = ['./yaml/ml-1m_lightkg.yaml']
elif args.dataset == "book-crossing":
    config_file_list = ['./yaml/book-crossing_lightkg.yaml']
elif args.dataset == "Amazon-book":
    config_file_list = ['./yaml/Amazon-book_lightkg.yaml']

# Initialize configuration
config = Config(model=LightKG, dataset=args.dataset, config_file_list=config_file_list, config_dict=config_dict)

# Set seed for reproducibility
init_seed(config['seed'], config['reproducibility'])

# Initialize logger
init_logger(config)
logger = getLogger()

# Create dataset and prepare data
data = create_dataset(config)
logger.info(data)  # Log dataset information
train_data, valid_data, test_data = data_preparation(config, data)

# Reset seed for model initialization
config['seed'] = 200
init_seed(config['seed'], config['reproducibility'])

# Initialize the model
model = LightKG(config=config, dataset=train_data._dataset).to(config['device'])

# Reset seed back to original for training
config['seed'] = 2020
init_seed(config['seed'], config['reproducibility'])

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
