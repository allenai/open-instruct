import gradio as gr
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

if len(sys.argv) > 1:
    model_name_or_path = sys.argv[1]
else:
    raise ValueError("Please provide a model name or path as the first argument")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

model.half().cuda()

def instruct(instruction):
    with torch.inference_mode():
        input_text = instruction
        input_ids = tokenizer.encode(input_text, return_tensors='pt').cuda()
        output_ids = model.generate(input_ids, max_length=1024)[0]
        output_str = tokenizer.decode(output_ids[input_ids.shape[-1]:])
        return output_str.strip()

demo = gr.Interface(
    fn=instruct,
    inputs=gr.Textbox(lines=10, placeholder="Enter your instruction here..."),
    outputs="text", 
    title="Demo for Open-Instruct", 
    description="Model name or path: " + model_name_or_path
)

demo.launch(share=True, server_port=7860)