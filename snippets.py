import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, pipeline
from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    print("ERROR: HF_TOKEN is not defined")

model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

# Load tokenizer and configure the text-generation pipeline with concise output
TOKENIZER = AutoTokenizer.from_pretrained(model_id)
text_gen = pipeline(
    'text-generation',
    model=model_id,
    tokenizer=TOKENIZER,
    torch_dtype=torch.bfloat16,
    device_map='auto',  # device_map={'': 0} gives out of memory on the GPU
    return_full_text=False,  # only return generated text, not the full prompt
)

# Prompt instructs the model to answer succinctly in one sentence without repetition
prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Answer the question in one clear sentence without repeating yourself.\n"
    "If the {animal} is on the table, where is the cat?\n"
    "Answer:"
)

# Configure model kwargs to enforce brevity and reduce repetition
llm = HuggingFacePipeline(
    pipeline=text_gen,
    model_kwargs={
        "max_new_tokens": 25,
        "pad_token_id": TOKENIZER.eos_token_id,
        "temperature": 0.0,
        "repetition_penalty": 1.2,
        "do_sample": False,
    },
)

# Compose prompt and llm using Runnable (pipe) to replace deprecated LLMChain
chain = prompt | llm

# Stop generation at the first newline to ensure single-sentence answer
result = chain.invoke({"animal": "cat"}, stop=["\n"])
print(result)
