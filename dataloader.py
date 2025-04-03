from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import datasets

def mantis_QA_loader(image_placeholder):
    dataset = datasets.load_dataset("TIGER-Lab/Mantis-Instruct", "multi_vqa", revision="script",
                                    cache_dir=".cache")
    
    for item in tqdm(dataset['train']):
        id = item['id']
        image_data = item['images']
        img_path = [image['path'] for image in image_data]
        img_files = [Image.open(path).convert("RGB").resize((224, 224)) for path in img_path]
        conversations = item['conversation']
        number_of_image = len(img_path)

        for i, conversation in enumerate(conversations):
            if i % 2 == 0: # user
                question = conversation['content']
                # remove the "<image>" placeholder from question
                question = question.replace("<image>", "") + image_placeholder * number_of_image
            else: # assistant
                answer = conversation['content']
                yield {
                    "id": id,
                    "question": question,
                    "image_files": img_files,
                    "answer": answer,
                    "num_image": number_of_image
                }
                break