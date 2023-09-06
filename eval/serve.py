import sys
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from model_loader import *
from fastapi import FastAPI
from pydantic import BaseModel, Field

model_name = '/models/Yarn-Llama-2-13b-128k'

tokenizer = AutoTokenizer.from_pretrained(
    model_name, model_max_length=sys.maxsize, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

from scaled_rope.modeling_llama_together_yarn import LlamaForCausalLM
from scaled_rope.configuration_llama import LlamaConfig
model_cls = LlamaForCausalLM
config_cls = LlamaConfig

config = config_cls.from_pretrained(model_name, trust_remote_code=True)

model = model_cls.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    config=config,
    quantization_config=None
).eval()

app = FastAPI(title=f"Serving {model_name}", version="0.1",)

class PromptLength(BaseModel):
    prompt:str = Field("You say you're Leo Tolstoy, but in reality", title='Model prompt')

@app.post("/get_n_token/")
def get_n_token(prompt:PromptLength):
    return {"n_token":  len(tokenizer.tokenize(prompt.prompt))}

class Prompt(BaseModel):
    prompt:str = Field("You say you're Leo Tolstoy, but in reality", title='Model prompt')
    max_new_tokens:int = Field(256, ge=1, le=1024*128, title='Number of tokens generated in each sample')
    temperature:float = Field(1.0, ge=0.1, le=10.0, title='Temperature parameter for generation')
    top_k:int = Field(40, ge=1, le=30000)
    repetition_penalty:float = Field(1.1, ge=1.0, )
    penalty_alpha:float = Field(0.0, ge=0.0, )
    num_return_sequences:int = Field(1, ge=1, le=5, title='Number of samples generated')

@app.post("/generate/")
def gen_sample(prompt: Prompt):
    with torch.no_grad():
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id,
                        temperature=prompt.temperature, repetition_penalty=prompt.repetition_penalty,
                        top_k=prompt.top_k, penalty_alpha=prompt.penalty_alpha, do_sample=prompt.temperature is not None)
    input_tokens = len(tokenizer.tokenize(prompt.prompt))
    if input_tokens + prompt.max_new_tokens > config.max_position_embeddings: 
        return {"error": f'N of input tokens ({input_tokens}) + prompt.max_new_tokens ({prompt.max_new_tokens}) > config.max_position_embeddings ({config.max_position_embeddings})'}

    return {"replies": pipe(prompt.prompt, num_return_sequences=1, max_new_tokens=prompt.max_new_tokens)[
            0]["generated_text"][len(prompt.prompt):]}

@app.get("/health")
def healthcheck():
    return True
