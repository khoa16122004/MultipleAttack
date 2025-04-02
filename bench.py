from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import torch
import torch.nn.functional as F


sim_model = SentenceTransformer('all-MiniLM-L6-v2')
@torch.no_grad()
def FreeText_benchmark(args, image_tensors, index_attack, input_ids, image_sizes, 
                       gt_answer, pertubation_list, model):
    
    # image_tensors: batch x 3 x W x H
    # pertubation_list: 3 x W x H
    
    
    adv_img_tensors = image_tensors.detach().clone().cuda()
    adv_img_tensors[index_attack] = image_tensors[index_attack] + pertubation_list
    adv_pil_images = model.decode_image_tensors(adv_img_tensors) # torch ten
    output = model.inference(input_ids, adv_img_tensors, image_sizes)[0]    
    
    # cosine similarity
    emb1 = sim_model.encode(output, convert_to_tensor=True)
    emb2 = sim_model.encode(gt_answer, convert_to_tensor=True)
    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    s1 = 0.1 - similarity
    
    # BLEU score
    bleu = sentence_bleu([gt_answer.split()], output.split())
    s2 = 0.1 - bleu
    
    # number of words
    num_words = len(output.split())
    s3 = 0.1 * (10 - num_words)
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(gt_answer, output)
    rouge1 = rouge_scores['rouge1'].fmeasure
    rouge2 = rouge_scores['rouge2'].fmeasure
    rougeL = rouge_scores['rougeL'].fmeasure
    s4 = 0.3 - ((rouge1 + rouge2 + rougeL)/3)
    
    
    # weighted sum
    # final_score = s1 + s2 + s3 + s4 + s5
    final_score = s1 + s2 + s3 +s4
    return final_score, adv_pil_images, output, adv_img_tensors



def FreeText_all_benchmark(args, image_tensors, index_attack, input_ids, image_sizes, 
                           gt_answer, pertubation_list, model):
    
    # image_tensors: batch x 3 x W x H
    # pertubation_list: 1 x batch x 3 x W x H
    
    
    adv_img_tensors = image_tensors.detach().clone().cuda()
    adv_img_tensors[index_attack] = image_tensors + pertubation_list
    adv_pil_images = model.decode_image_tensors(adv_img_tensors) # torch ten
    output = model.inference(input_ids, adv_img_tensors, image_sizes)[0]    
    
    # cosine similarity
    emb1 = sim_model.encode(output, convert_to_tensor=True)
    emb2 = sim_model.encode(gt_answer, convert_to_tensor=True)
    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    s1 = 0.1 - similarity
    
    # BLEU score
    bleu = sentence_bleu([gt_answer.split()], output.split())
    s2 = 0.1 - bleu
    
    # number of words
    num_words = len(output.split())
    s3 = 0.1 * (10 - num_words)
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(gt_answer, output)
    rouge1 = rouge_scores['rouge1'].fmeasure
    rouge2 = rouge_scores['rouge2'].fmeasure
    rougeL = rouge_scores['rougeL'].fmeasure
    s4 = 0.3 - ((rouge1 + rouge2 + rougeL)/3)
    
    
    # weighted sum
    # final_score = s1 + s2 + s3 + s4 + s5
    final_score = s1 + s2 + s3 +s4
    return final_score, adv_pil_images, output, adv_img_tensors