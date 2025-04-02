from models.llava_ import LLava
from PIL import Image
import os
import torch
clean_image_dir = "/data/elo/khoatn/MultipleAttack/test_ES_lambda=50_epsilon=0.01_maxiter=1_pretrained=llava-onevision-qwen2-7b-ov/30/clean_img"
output_dir = "verify"
os.makedirs(output_dir, exist_ok=True)
model = LLava("llava-onevision-qwen2-7b-ov", "llava_qwen")

qs = "Which images feature elements typically associated with celebration or festive occasions?<image><image><image><image>"
# kiểm định xem ảnh float inference có khác ảnh int ko
img_files = [Image.open(os.path.join(clean_image_dir, f"{j}.png")).convert("RGB") for j in range(len(os.listdir(clean_image_dir)))]
adv_img_file = Image.open(r"/data/elo/khoatn/MultipleAttack/test_ES_lambda=50_epsilon=0.01_maxiter=1_pretrained=llava-onevision-qwen2-7b-ov/30/0/adv.png")
img_files[0] = adv_img_file
print("Size: ", adv_img_file.size)
img_files[0].save("testadv.png")
input_ids, image_tensors_0, image_sizes = model.repair_input(qs, img_files)
print("imgsize: ", image_sizes)
tensor_output = model.inference(input_ids, image_tensors_0, image_sizes)
print("PIL inference: ", tensor_output)



image_tensor_load = torch.load(r"/data/elo/khoatn/MultipleAttack/test_ES_lambda=50_epsilon=0.01_maxiter=1_pretrained=llava-onevision-qwen2-7b-ov/30/0/all_adv.pt")
tensor_output = model.inference(input_ids, image_tensor_load, image_sizes)
print("Tensor inference: ", tensor_output)

print("diff: ", (image_tensors_0[0] - image_tensor_load[0]).abs().sum())


# decoded_image_pil = model.decode_image_tensors(image_tensors_0) # ảnh int

# input_ids, image_tensors_1, image_sizes = model.repair_input("Descibe image<image><image>", decoded_image_pil)
# tensor_output = model.inference(input_ids, image_tensors_1, image_sizes)
# for i in range(len(decoded_image_pil)):
#     decoded_image_pil[i].save(os.path.join(output_dir, f"decoded_image{i}.png"))
#     # img_files[i].save(os.path.join(output_dir, f"original_image{i}.png"))
    