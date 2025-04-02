from utils import seed_everything, init_model
from dataloader import mantis_QA_loader
from attack import ES_1_lambda
from bench import FreeText_benchmark
import argparse
import os
import torch

def main(args):
    seed_everything(22520691)
    model, image_token, special_token = init_model(args)
    
    # repair dir
    experiment_dir = f"{args.prefix_path}_ES_lambda={args.lambda_}_epsilon={args.epsilon}_maxiter={args.max_query}_pretrained={args.pretrained}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    
    for i, item in enumerate(mantis_QA_loader(image_placeholder=image_token)):

        
        if i > args.index_sample:
            break
        
        if i != args.index_sample:
            continue
        

        
        # repair dir
        sample_dir = os.path.join(experiment_dir, str(i))
        os.makedirs(sample_dir, exist_ok=True)
        
        # take information
        qs, img_files, gt_answer, num_image = item['question'], item['image_files'], item['answer'], item['num_image'] 
        
        # 
        input_ids, image_tensors, image_sizes = model.repair_input(qs, img_files)
        print("Image size: ", image_tensors.shape)
        
        # repair dir
        clean_dir = os.path.join(sample_dir, "clean_img")
        os.makedirs(clean_dir, exist_ok=True)
        for j, img_clean_files in enumerate(img_files):
            img_clean_files.save(os.path.join(clean_dir, f"{j}.png"))
        
        # inference
        original_output = model.inference(input_ids, image_tensors, image_sizes)[0]
        print("Question: ", qs)
        print("Original output: ", original_output)
        print("Ground truth answer: ", gt_answer)

        for index_attack in range(num_image):
            # repair dir
            index_dir = os.path.join(sample_dir, str(index_attack))
            os.makedirs(index_dir, exist_ok=True)
            
            
            num_evaluation, history, best_img_files_adv, success, output, best_adv_img_tensors =ES_1_lambda(args, FreeText_benchmark, index_attack, model, args.lambda_,
                                                                                                            image_tensors, image_sizes, input_ids, original_output, 
                                                                                                            epsilon=args.epsilon)
            
            # log
            attacked_img_files = best_img_files_adv[index_attack]
            attacked_img_files.save(os.path.join(index_dir, "adv.png"))
            torch.save(best_adv_img_tensors, os.path.join(index_dir, "all_adv.pt"))
            
            with open(os.path.join(index_dir, "history.txt"), "w") as f:          
                for i, fitness in enumerate(history):
                    f.write(f"iteration {i}: {fitness}\n")
            
            with open(os.path.join(index_dir, "output.txt"), "w") as f:
                f.write(f"Question: {qs}\n\n")
                f.write(f"Ground truth answer: {gt_answer}\n\n")
                f.write(f"Original output: {original_output}\n\n")
                f.write(f"Attacked output: {output}\n\n")
                f.write(f"Fitness:  {history[-1]}",)
                f.write(f"Num evaluation: {num_evaluation}\n\n")
                        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_sample", type=int, default=0)
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--max_query", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--run", type=int, default=10)
    parser.add_argument("--lambda_", type=int, default=50)
    parser.add_argument("--prefix_path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="mantis_qa")
    args = parser.parse_args()

    main(args)