import random
from random import randint
import torch
from tqdm import tqdm 
import numpy as np
import os
import pandas as pd
import copy
import numpy as np
import torch
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
from recbole.config import Config
#from dataloader import data_pre
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from LightKG import *
#from mul_lightkg import *
from CFKG import *
from KGAT import *
from KGRec import *
from CFKG import *
from CKE import *
from RippleNet import *
from KGIN import *
from KGCN import *
from LightGCN import *
from MCCLK import *
import torch.profiler as profiler
from recbole.trainer import KGTrainer,KGATTrainer
import torch.nn.functional as F
from recbole.quick_start.quick_start import load_data_and_model

config_dict = {'seed':2020}  
config_file_list=['./yaml/ml-1m_LightKG.yaml']

config = Config(model=LightKG, dataset='ml-1m', config_file_list=config_file_list, config_dict=config_dict)
init_seed(config['seed'], config['reproducibility']) 
init_logger(config)
logger = getLogger()
data = create_dataset(config)
logger.info(data)
train_data, valid_data, test_data = data_preparation(config,data)

config['seed'] = 200
init_seed(config['seed'], config['reproducibility']) 
model = LightKG(config=config, dataset=train_data._dataset).to(config['device'])
config['seed'] = 2020
init_seed(config['seed'], config['reproducibility']) 
#torch.autograd.set_detect_anomaly(True)
logger.info(model)
trainer = KGTrainer(config, model)

best_valid_score, best_valid_result = trainer.fit(train_data, valid_data) 
test_result = trainer.evaluate(test_data)

logger.info('best valid result: {}'.format(best_valid_result))
logger.info('test result: {}'.format(test_result))
