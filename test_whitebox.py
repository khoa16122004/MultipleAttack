from utils import seed_everything, init_model
from dataloader import mantis_QA_loader
from attack import ES_1_each_lambda, ES_1_all_lambda
from bench import FreeText_benchmark, FreeText_all_benchmark
import argparse
import os
import torch
import torchvision.transforms as transforms


def main(args):
    seed_everything(22520691)
    model, image_token, special_token = init_model(args)
    
    
    for i, item in enumerate(mantis_QA_loader(image_placeholder=image_token)):
        
        # take information
        qs, img_files, gt_answer, num_image = item['question'], item['image_files'], item['answer'], item['num_image'] 
                
        # inference
        input_ids, image_tensors, image_sizes = model.repair_input(qs, img_files)
        model.model(input_ids=input_ids, images=image_tensors, image_sizes=image_sizes)
        break
    
     


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--dataset", type=str, default="mantis_qa")
    args = parser.parse_args()

    main(args)