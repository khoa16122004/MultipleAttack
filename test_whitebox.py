from utils import seed_everything, init_model
from dataloader import mantis_QA_loader
from attack import ES_1_each_lambda, ES_1_all_lambda
from bench import FreeText_benchmark, FreeText_all_benchmark
import argparse
import os
import torch
import torchvision.transforms as transforms


def pgd_attack(model, input_ids, image_tensors, image_sizes, target_answer, epsilon=0.01, alpha=0.005, num_steps=20):
    # Bật requires_grad cho image_tensors để tính gradient
    image_tensors.requires_grad_(True)
    
    # Sao chép ảnh gốc để làm ảnh tấn công
    attacked_image = image_tensors.clone()
    
    for _ in range(num_steps):
        # Chạy mô hình để lấy logits
        logits = model.model(input_ids=input_ids, images=attacked_image, image_sizes=image_sizes, dpo_forward=True)
        
        # Tính toán loss giữa logits và target_answer
        # Chuyển target_answer thành tensor (mã hóa thành token ids)
        target_ids = model.tokenizer.encode(target_answer, return_tensors="pt").cuda()

        # Tính toán cross-entropy loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        # Tính gradient của loss đối với ảnh tấn công
        model.zero_grad()
        loss.backward()

        # Lấy gradient của ảnh tấn công
        grad = attacked_image.grad

        # Cập nhật ảnh theo hướng của gradient (PGD)
        attacked_image = attacked_image + alpha * grad.sign()

        # Giới hạn ảnh tấn công trong phạm vi hợp lý
        attacked_image = torch.clamp(attacked_image, 0, 1)

        # Đặt lại gradient về 0 sau mỗi bước
        attacked_image.grad.zero_()

    # Trả về ảnh tấn công
    return attacked_image

def main(args):
    seed_everything(22520691)
    model, image_token, special_token = init_model(args)
    
    for i, item in enumerate(mantis_QA_loader(image_placeholder=image_token)):
        
        # Lấy thông tin từ dữ liệu
        qs, img_files, gt_answer, num_image = item['question'], item['image_files'], item['answer'], item['num_image'] 
        target_answer = "I don't know!"  # target_answer mà bạn muốn mô hình trả về
        
        # Inference: chuẩn bị input và image tensors
        input_ids, image_tensors, image_sizes = model.repair_input(qs, img_files)
        
        # Thực hiện tấn công PGD
        attacked_image = pgd_attack(model, input_ids, image_tensors, image_sizes, target_answer)

        # Bây giờ bạn có thể sử dụng attacked_image cho các mục đích tiếp theo (ví dụ: inference, evaluation, ...)

        print("Đã tạo ảnh tấn công:", attacked_image)  # In ảnh tấn công (tensor bị tấn công)
        break
    
     


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--dataset", type=str, default="mantis_qa")
    args = parser.parse_args()

    main(args)