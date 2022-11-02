import torch
from Conferences.IGN_CF.igcncf_github.utils import init_run

from dataset import get_dataset
from Conferences.IGN_CF.igcncf_github.config import get_gowalla_config, get_yelp_config, get_amazon_config
from tensorboardX import SummaryWriter
import os, sys

###THIS CODE IS FROM run.py FROM ORIGINAL IMPLEMENTATION
def init_file_and_device():
    log_path = __file__[:-3]
    init_run(log_path, 2021)
    device = torch.device('cpu')
    return device, log_path


def acquire_dataset(log_path, config):
    dataset_config, model_config, trainer_config = config[2]
    #dataset_config['path'] = dataset_config['path'][:-4] + str(1)
    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config)
    return dataset


def txt_to_csv(folder):
    for filename in os.listdir(folder):
        infilename = os.path.join(folder, filename)
        if not os.path.isfile(infilename): continue
        oldbase = os.path.splitext(filename)
        newname = infilename.replace('.txt', '.csv')
        os.rename(infilename, newname)
