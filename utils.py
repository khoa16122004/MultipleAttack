import random, os
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def init_model(args):
    special_token = None
    if "llava" in args.model_name:
        from models.llava_ import LLava
        image_token = "<image>"
        model = LLava(args.pretrained, args.model_name)

    elif "openflamingo" in args.model_name:
        from models.openflamingo_ import OpenFlamingo
        image_token = "<image>"
        special_token = "<|endofchunk|>"

        model = OpenFlamingo(args.pretrained)
    
    elif "mantis" in args.model_name:
        from models.mantis_ import Mantis
        image_token = "<image>"
        model = Mantis(args.pretrained)
        
    elif "deepseek" in args.model_name:
        from models.deepseek_ import DeepSeek
        image_token = "<image_placeholder>"
        model = DeepSeek(args.pretrained)
    
    return model, image_token, special_token


        

