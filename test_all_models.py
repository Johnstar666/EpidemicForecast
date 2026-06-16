#!/usr/bin/env python3
import os
import subprocess
import sys

def extract_model_info(model_dir):
    """Extract country, model_name, and learning_rate from model directory name"""
    # The format is: epi_forcasting_{country}_{model_id}_EpiLLm_{model_type}_epidataset_...
    
    if 'Italy' in model_dir:
        country = 'italy'
        node_num = 105
        root_path = '/root/experiment/EpiLLM/dataset/Italy'
        data_path = 'italy_labels.csv'
    elif 'England' in model_dir:
        country = 'england'
        node_num = 129
        root_path = '/root/experiment/EpiLLM/dataset/England'
        data_path = 'england_labels.csv'
    else:
        return None, None, None, None, None, None, None, None
    
    # Extract model type (Gpt2, gemma1, or DS3)
    if 'EpiLLm_Gpt2' in model_dir:
        model_name = 'EpiLLm_Gpt2'
        llm_ckp_dir = '/root/autodl-tmp/models/gpt2'
    elif 'EpiLLm_gemma1' in model_dir:
        model_name = 'EpiLLm_gemma1'
        llm_ckp_dir = '/root/autodl-tmp/models/gemma-3-1b-pt'
    elif 'EpiLLm_DS3' in model_dir:
        model_name = 'EpiLLm_DS3'
        llm_ckp_dir = '/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B'
    else:
        return None, None, None, None, None, None, None, None
    
    # Extract learning rate
    lr_start = model_dir.find('lr')
    if lr_start != -1:
        lr_end = model_dir.find('_bt', lr_start)
        if lr_end == -1:
            lr_end = model_dir.find('_wd', lr_start)
        if lr_end == -1:
            # Take until the end or next underscore
            lr_part = model_dir[lr_start+2:]
            if '_' in lr_part:
                lr_part = lr_part.split('_')[0]
            learning_rate = float(lr_part)
        else:
            learning_rate = float(model_dir[lr_start+2:lr_end])
    else:
        learning_rate = 0.001  # default
    
    # Extract dataset/model_id (Italy_7d or England_7d)
    if country == 'italy':
        model_id = 'Italy_7d'
    else:
        model_id = 'England_7d'
    
    return country, model_name, learning_rate, node_num, root_path, data_path, llm_ckp_dir, model_id

def main():
    # Change to the EpiLLM directory
    os.chdir('/root/experiment/EpiLLM')
    
    # Set environment variables
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    
    # Get all checkpoint directories that contain "full_training_0"
    checkpoints_dir = './checkpoints'
    model_dirs = []
    
    for item in os.listdir(checkpoints_dir):
        if os.path.isdir(os.path.join(checkpoints_dir, item)) and 'training_0' in item and 'full_training_0' in item:
            model_dirs.append(item)
    
    print(f"Found {len(model_dirs)} models to test:")
    for model_dir in model_dirs:
        print(f"  - {model_dir}")
    
    # Test each model
    for i, model_dir in enumerate(model_dirs):
        print(f"\n[{i+1}/{len(model_dirs)}] Testing model: {model_dir}")
        
        # Extract model information
        result = extract_model_info(model_dir)
        if result[0] is None:
            print(f"  ⚠ Could not extract model info, skipping...")
            continue
            
        country, model_name, learning_rate, node_num, root_path, data_path, llm_ckp_dir, model_id = result
        
        print(f"  Country: {country}, Model: {model_name}, LR: {learning_rate}")
        
        # Build the test command
        cmd = (
            f"python run.py "
            f"--model {model_name} "
            f"--data epidataset "
            f"--root_path '{root_path}' "
            f"--data_path {data_path} "
            f"--node_num {node_num} "
            f"--llm_ckp_dir '{llm_ckp_dir}' "
            f"--model_id {model_id} "
            f"--seq_len 21 "
            f"--label_len 14 "
            f"--token_len 7 "
            f"--learning_rate {learning_rate} "
            f"--batch_size 1 "
            f"--is_training 0 "
            f"--visualize "
            f"--test_dir '{model_dir}' "
            f"--test_file_name checkpoint.pth"
        )
        
        # Activate conda environment and run command
        full_cmd = f"source /root/miniconda3/bin/activate epillm && cd /root/experiment/EpiLLM && {cmd}"
        
        print(f"  Running test...")
        
        try:
            # Run the command
            result = subprocess.run(
                ['bash', '-c', full_cmd],
                env=env,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                print(f"  ✓ Successfully tested {model_dir}")
                # Check if visualization was generated
                viz_path = f'./test_results/{model_dir}/7/0.png'
                if os.path.exists(viz_path):
                    print(f"    ✓ Visualization generated at {viz_path}")
                else:
                    print(f"    ⚠ No visualization found at {viz_path}")
            else:
                print(f"  ✗ Failed to test {model_dir}")
                print(f"    Error: {result.stderr[-500:]}")  # Show last 500 chars of error
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout while testing {model_dir}")
        except Exception as e:
            print(f"  ✗ Exception while testing {model_dir}: {e}")
    
    print(f"\nCompleted testing {len(model_dirs)} models.")
    print("Check the test_results directory for visualization images.")

if __name__ == "__main__":
    main()