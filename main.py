import torch
from src import linear_regress


linear_regress_model_param = linear_regress.train()
torch.save(linear_regress_model_param, 'model/linear_regress_model_param.pkl')
