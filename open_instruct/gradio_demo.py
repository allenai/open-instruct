# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
        output_ids = model.generate(input_ids, max_length=1024)[0]
        output_str = tokenizer.decode(output_ids[input_ids.shape[-1] :])
        return output_str.strip()


demo = gr.Interface(
    fn=instruct,
    inputs=gr.Textbox(lines=10, placeholder="Enter your instruction here..."),
    outputs="text",
    title="Demo for Open-Instruct",
    description="Model name or path: " + model_name_or_path,
)

demo.launch(share=True, server_port=7860)
