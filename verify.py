from models.llava_ import LLava
from PIL import Image
import os
clean_image_dir = "/data/elo/khoatn/MRAG-Bench/eval/models/ES_lambda=50_epsilon=0.1_maxiter=30_pretrained=llava-onevision-qwen2-7b-ov/0/clean_img"
output_dir = "verify"
os.makedirs(output_dir, exist_ok=True)
model = LLava("llava-onevision-qwen2-7b-ov", "llava_qwen")

img_files = [Image.open(os.path.listdir(clean_image_dir, path)).convert("RGB") for path in sorted(os.listdir(clean_image_dir))]
input_ids, image_tensors, image_sizes = model.repair_input(None, img_files)

decoded_image = model.decode_image_tensors(image_tensors)
for i in range(len(decoded_image)):
    decoded_image[i].save(os.path.join(output_dir, f"decoded_image{i}.png"))
    img_files[i].save(os.path.join(output_dir, f"original_image{i}.png"))
    