import torch
from tqdm import tqdm

def ES_1_each_lambda(args, benchmark, index_attack, model, lambda_,
                     image_tensors, image_sizes, input_ids, gt_answer, 
                     epsilon=0.05, sigma=1.5, c_increase=1.1, c_decrease=0.9,
                     verbose=True):
    
    # image_tensors: batch_size x 3 x 224 x 224
    best_pertubations = torch.randn_like(image_tensors[index_attack]).cuda()
    best_pertubations = torch.clamp(best_pertubations, -epsilon, epsilon)

    best_fitness, adv_img_files, output, best_adv_img_tensors = benchmark(args, image_tensors, index_attack, input_ids, image_sizes, 
                                                                          gt_answer, best_pertubations, model)
    best_img_files_adv = adv_img_files
    history = [best_fitness]
    success = False
    num_evaluation = 1
    
    for i in tqdm(range(args.max_query)):
        alpha = torch.randn(lambda_, *image_tensors[index_attack].shape).to(torch.float16).cuda()
        pertubations_list = alpha + best_pertubations * sigma
        pertubations_list = torch.clamp(pertubations_list, -epsilon, epsilon)
        
        # inference
        current_fitnesses = []
        current_adv_files = []
        current_outputs = []
        current_adv_img_tensors = []
        for pertubations in pertubations_list:
            fitness, adv_img_files, current_output, adv_img_tensor = benchmark(args, image_tensors, index_attack, input_ids, image_sizes, 
                                                                       gt_answer, pertubations, model)    
            

            current_fitnesses.append(fitness)
            current_adv_files.append(adv_img_files)
            current_outputs.append(current_output)
            current_adv_img_tensors.append(adv_img_tensor)
        
        num_evaluation += lambda_
        print("Current fitness: ", current_fitnesses)
        # print("Current output: ", current_outputs)
            
        current_fitnesses = torch.tensor(current_fitnesses)
        best_id_current_fitness = torch.argmax(current_fitnesses) 
        
        if current_fitnesses[best_id_current_fitness] >  best_fitness:
            best_fitness = current_fitnesses[best_id_current_fitness]
            best_pertubations = pertubations_list[best_id_current_fitness]
            best_img_files_adv = current_adv_files[best_id_current_fitness]
            output = current_outputs[best_id_current_fitness]
            best_adv_img_tensors = current_adv_img_tensors[best_id_current_fitness]
            sigma *= c_increase
        else:
            sigma *= c_decrease
            
        history.append(best_fitness)
        
        
        if verbose == True:
            print(f"Iteration {i}, best fitness: {best_fitness}, output: {output}\n")
        
    return num_evaluation, history, best_img_files_adv, success, output, best_adv_img_tensors
    
    
def ES_1_all_lambda(args, benchmark, model, lambda_,
                     image_tensors, image_sizes, input_ids, gt_answer, 
                     epsilon=0.05, sigma=1.5, c_increase=1.1, c_decrease=0.9,
                     verbose=True):
    
    # image_tensors: batch_size x 3 x 224 x 224
    best_pertubations = torch.randn_like(image_tensors).cuda()
    best_pertubations = torch.clamp(best_pertubations, -epsilon, epsilon)

    best_fitness, adv_img_files, output, best_adv_img_tensors = benchmark(args, image_tensors, input_ids, image_sizes, 
                                                                          gt_answer, best_pertubations, model)
    best_img_files_adv = adv_img_files
    history = [best_fitness]
    success = False
    num_evaluation = 1
    
    for i in tqdm(range(args.max_query)):
        alpha = torch.randn(lambda_, *image_tensors.shape).to(torch.float16).cuda()
        pertubations_list = alpha + best_pertubations * sigma
        pertubations_list = torch.clamp(pertubations_list, -epsilon, epsilon)
        
        # inference
        current_fitnesses = []
        current_adv_files = []
        current_outputs = []
        current_adv_img_tensors = []
        for pertubations in pertubations_list:
            fitness, adv_img_files, current_output, adv_img_tensor = benchmark(args, image_tensors, input_ids, image_sizes, 
                                                                               gt_answer, pertubations, model)    
            

            current_fitnesses.append(fitness)
            current_adv_files.append(adv_img_files)
            current_outputs.append(current_output)
            current_adv_img_tensors.append(adv_img_tensor)
        
        num_evaluation += lambda_
        print("Current fitness: ", current_fitnesses)
        # print("Current output: ", current_outputs)
            
        current_fitnesses = torch.tensor(current_fitnesses)
        best_id_current_fitness = torch.argmax(current_fitnesses) 
        
        if current_fitnesses[best_id_current_fitness] >  best_fitness:
            best_fitness = current_fitnesses[best_id_current_fitness]
            best_pertubations = pertubations_list[best_id_current_fitness]
            best_img_files_adv = current_adv_files[best_id_current_fitness]
            output = current_outputs[best_id_current_fitness]
            best_adv_img_tensors = current_adv_img_tensors[best_id_current_fitness]
            sigma *= c_increase
        else:
            sigma *= c_decrease
            
        history.append(best_fitness)
        
        
        if verbose == True:
            print(f"Iteration {i}, best fitness: {best_fitness}, output: {output}\n")
        
    return num_evaluation, history, best_img_files_adv, success, output, best_adv_img_tensors
    
    