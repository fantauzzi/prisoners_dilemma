import os
import transformers
import torch
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    print("ERROR: HF_TOKEN is not defined")

model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

pipeline = transformers.pipeline('text-generation',
                                 model=model_id,
                                 model_kwargs={'torch_dtype': torch.bfloat16},
                                 device_map='auto',
                                 max_length=200)

from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

prompt = PromptTemplate.from_template('If the cat is on the table, where is the cat? Is the cat under the table?')
llm = HuggingFacePipeline(pipeline=pipeline)
chain = LLMChain(llm=llm, prompt=prompt)
chain.run()



