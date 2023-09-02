# If you want to include system prompt, see this discussion for the template: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/discussions/4
# However, see here that removing the system prompt actually reduce the false refusal rates: https://github.com/facebookresearch/llama/blob/main/UPDATES.md?utm_source=twitter&utm_medium=organic_social&utm_campaign=llama2&utm_content=text#observed-issue
llama2_prompting_template = '''[INST] {prompt} [/INST]'''

# See the tulu template here: https://huggingface.co/allenai/tulu-7b#input-format
tulu_prompting_template = '''<|user|>\n{prompt}\n<|assistant|>\n'''