from llava_ import LLava
from PIL import Image
import os
import torch

model = LLava("llava-onevision-qwen2-7b-ov", "llava_qwen")
qs = "Which images feature elements typically associated with celebration or festive occasions?<image><image><image><image>"
print(model.model)