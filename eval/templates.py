
def create_prompt_with_tulu_chat_format(messages, tokenizer, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text

# weirdness with olmo tokenizer means IP_ADDR is the eos token.
def create_prompt_with_olmo_chat_format(messages, tokenizer, bos="|||IP_ADDRESS|||", eos="|||IP_ADDRESS|||", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Olmo chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text  # forcibly add bos
    return formatted_text


def create_prompt_with_llama2_chat_format(messages, tokenizer, bos="<s>", eos="</s>", add_bos=True):
    '''
    This function is adapted from the official llama2 chat completion script: 
    https://github.com/facebookresearch/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/generation.py#L274
    '''
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    B_INST, E_INST = "[INST]", "[/INST]"
    formatted_text = ""
    # If you want to include system prompt, see this discussion for the template: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/discussions/4
    # However, see here that removing the system prompt actually reduce the false refusal rates: https://github.com/facebookresearch/llama/blob/main/UPDATES.md?utm_source=twitter&utm_medium=organic_social&utm_campaign=llama2&utm_content=text#observed-issue
    if messages[0]["role"] == "system":
        assert len(messages) >= 2 and messages[1]["role"] == "user", "LLaMa2 chat cannot start with a single system message."
        messages = [{
            "role": "user",
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"]
        }] + messages[2:]
    for message in messages:
        if message["role"] == "user":
            formatted_text += bos + f"{B_INST} {(message['content']).strip()} {E_INST}"
        elif message["role"] == "assistant":
            formatted_text += f" {(message['content'])} " + eos
        else:
            raise ValueError(
                "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    # The llama2 chat template by default has a bos token at the start of each user message.
    # The next line removes the bos token if add_bos is False.
    formatted_text = formatted_text[len(bos):] if not add_bos else formatted_text
    return formatted_text


def create_prompt_with_xwin_chat_format(messages, tokenizer, bos="<s>", eos="</s>", add_bos=True):
    '''
    This function is adapted from the official xwin chat completion script:
    https://huggingface.co/Xwin-LM/Xwin-LM-70B-V0.1
    '''
    formatted_text = "A chat between a curious user and an artificial intelligence assistant. "
    formatted_text += "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    for message in messages:
        if message["role"] == "user":
            formatted_text += "USER: " + message["content"] + " "
        elif message["role"] == "assistant":
            formatted_text += "ASSISTANT: " + message["content"] + eos
    formatted_text += "ASSISTANT:"
    return formatted_text


def create_prompt_with_zephyr_chat_format(messages, tokenizer, bos="<s>", eos="</s>", add_bos=True):
    '''
    This function is adapted from the official zephyr chat completion script:
    https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
    '''
    formatted_text = ""
    # if messages[0]["role"] != "system":
    #     messages = [{
    #         "role": "system",
    #         "content": ""
    #     }] + messages

    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + eos + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + eos + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"] + eos + "\n"
        else:
            raise ValueError(
                "Zephyr chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    formatted_text += "<|assistant|>\n"
    return formatted_text    

# helper for just using the huggingface tokenizer
def create_prompt_with_huggingface_tokenizer_template(messages, tokenizer, add_bos=False):
    formatted_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if add_bos:
        formatted_text = tokenizer.bos_token + formatted_text
    return formatted_text
