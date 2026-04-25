# EpiLLM

This is the implementation of EpiLLM.

## Usage

1. Install Pytorch and necessary dependencies.

   ``` 
   pip install -r requirements.txt
   ```

2. Download the large language models from [Hugging Face](https://huggingface.co/). The default LLM is GPT2. you can change the `llm_ckp_dir` in `run.py` to use other LLMs.

   * [gpt2](https://huggingface.co/openai-community/gpt2)
   * [gemma-3](https://huggingface.co/google/gemma-3-1b-pt)

   * [DeepSeek-R1-Qwen-1.5B ](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

   You can create a folder (e.g. named `gpt`) in the root directory of your code and place the weight files inside for use.
