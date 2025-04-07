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
        
        # Bật requires_grad cho image_tensors
        image_tensors.requires_grad_(True)
        
        # forward pass (chạy mô hình để lấy logits)
        logits = model.model(input_ids=input_ids, images=image_tensors, image_sizes=image_sizes, dpo_forward=True)
        
        # Giả sử bạn có loss function (ví dụ cross entropy loss)
        # Giả sử bạn có ground truth (gt_answer) đã chuyển thành tensor phù hợp
        target = torch.tensor(gt_answer).to(logits.device)  # Chuyển ground truth thành tensor
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        
        # Tính gradient
        loss.backward()
        
        # Lấy gradient của image_tensors
        image_grads = image_tensors.grad
        
        # Dùng gradient của image_tensors cho mục đích tiếp theo (VD: adversarial attack)
        print(image_grads)  # Hoặc xử lý ảnh theo gradient nếu cần
        break
    
     


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--dataset", type=str, default="mantis_qa")
    args = parser.parse_args()

    main(args)