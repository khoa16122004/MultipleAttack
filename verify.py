from models.llava_ import LLava
from PIL import Image
import os
import torch
clean_image_dir = "/data/elo/khoatn/MRAG-Bench/eval/models/ES_lambda=50_epsilon=0.1_maxiter=30_pretrained=llava-onevision-qwen2-7b-ov/0/clean_img"
output_dir = "verify"
os.makedirs(output_dir, exist_ok=True)
model = LLava("llava-onevision-qwen2-7b-ov", "llava_qwen")


# kiểm định xem ảnh float inference có khác ảnh int ko
img_files = [Image.open(os.path.join(clean_image_dir, path)).convert("RGB").resize((224, 224)) for path in sorted(os.listdir(clean_image_dir))]
input_ids, image_tensors, image_sizes = model.repair_input(None, img_files)
image_tensors += torch.randn_like(image_tensors).cuda() * 0.05 # ảnh float
tensor_output = model.inference(input_ids, image_tensors, image_sizes)
print(tensor_output)


decoded_image_pil = model.decode_image_tensors(image_tensors) # ảnh int

input_ids, image_tensors, image_sizes = model.repair_input(None, decoded_image_pil)
tensor_output = model.inference(input_ids, image_tensors, image_sizes)
for i in range(len(decoded_image_pil)):
    decoded_image_pil[i].save(os.path.join(output_dir, f"decoded_image{i}.png"))
    # img_files[i].save(os.path.join(output_dir, f"original_image{i}.png"))
    