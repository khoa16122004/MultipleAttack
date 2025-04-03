from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import torch
import torch.nn.functional as F
import warnings
import torchvision.transforms as transforms

warnings.filterwarnings("ignore", category=UserWarning)
topil = transforms.ToPILImage()

sim_model = SentenceTransformer('all-MiniLM-L6-v2')
@torch.no_grad()
def FreeText_benchmark(args, image_tensors, index_attack, input_ids, image_sizes, 
                       gt_answer, pertubation_list, model,
                       target_answer="I dont know!"):
    
    # image_tensors: batch x 3 x W x H
    # pertubation_list: 3 x W x H
    
    
    adv_img_tensors = image_tensors.detach().clone().cuda()
    adv_img_tensors[index_attack] = image_tensors[index_attack] + pertubation_list
    adv_pil_images = model.decode_image_tensors(adv_img_tensors) # torch ten
    output = model.inference(input_ids, adv_img_tensors, image_sizes)[0]    
    
    # cosine similarity
    emb1 = sim_model.encode(output, convert_to_tensor=True)
    emb2 = sim_model.encode(target_answer, convert_to_tensor=True)
    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    # print("Embedding similarity: ", similarity)
    
    # BLEU score
    bleu = sentence_bleu([target_answer.split()], output.split())
    # print("Bleu: ", bleu)
    
    # number of words
    num_words = 0.01 * len(output.split())
    # print("Num word: ", num_words)
    
    # weighted sum
    # final_score = s1 + s2 + s3 + s4 + s5
    final_score = (similarity + bleu + num_words) / 3
    return final_score, adv_pil_images, output, adv_img_tensors



def FreeText_all_benchmark(args, image_tensors, input_ids, image_sizes, 
                           gt_answer, pertubation_list, model,
                           target_answer="I dont know!"):
    
    # image_tensors: batch x 3 x W x H
    # pertubation_list: 1 x batch x 3 x W x H
    
    
    adv_img_tensors = image_tensors.detach().clone().cuda()
    adv_img_tensors = image_tensors + pertubation_list
    pil_adv_imgs = [topil(adv_img_tensor) for adv_img_tensor in adv_img_tensors]
    print(pil_adv_imgs)
    _, adv_img_tensors, _ = model.repair_input(None, adv_img_tensors)
        
    output = model.inference(input_ids, adv_img_tensors, image_sizes)[0]    
    
    # cosine similarity
    emb1 = sim_model.encode(output, convert_to_tensor=True)
    emb2 = sim_model.encode(gt_answer, convert_to_tensor=True)
    emb3 = sim_model.encode(target_answer, convert_to_tensor=True)
    similarity_1 = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item() # sim with gt answer
    similarity_2 = F.cosine_similarity(emb1.unsqueeze(0), emb3.unsqueeze(0)).item() # sim with target
    
    # print("sim gt and output: ", similarity_1)
    # print("Embedding tg and output: ", similarity_2)
    
    # # BLEU score
    # bleu = sentence_bleu([target_answer.split()], output.split())
    # # print("Bleu: ", bleu)
    
    # number of words
    num_words = 0.01 * len(output.split())
    # print("Num word: ", num_words)
    
    # weighted sum
    # final_score = s1 + s2 + s3 + s4 + s5
    # final_score = (similarity + bleu + num_words) / 3
    final_score = (similarity_2 - similarity_1 - num_words) / 2
    return final_score, pil_adv_imgs, output, adv_img_tensors