import gradio as gr
import torch
import sys
import html
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

if len(sys.argv) > 1:
    model_name_or_path = sys.argv[1]
else:
    raise ValueError("Please provide a model name or path as the first argument")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

model.half().cuda()

def convert_message(message):
    message_text = ""
    if message["content"] is None and message["role"] == "assistant":
        message_text += "<|assistant|>\n"  # final msg
    elif message["role"] == "system":
        message_text += "<|system|>\n" + message["content"].strip() + "\n"
    elif message["role"] == "user":
        message_text += "<|user|>\n" + message["content"].strip() + "\n"
    elif message["role"] == "assistant":
        message_text += "<|assistant|>\n" + message["content"].strip() + "\n"
    else:
        raise ValueError("Invalid role: {}".format(message["role"]))
    # gradio cleaning - it converts stuff to html entities
    # we would need special handling for where we want to keep the html...
    message_text = html.unescape(message_text)
    # it also converts newlines to <br>, undo this.
    message_text = message_text.replace("<br>", "\n")
    return message_text

def convert_history(chat_history, max_input_length=1024):
    history_text = ""
    idx = len(chat_history) - 1
    # add messages in reverse order until we hit max_input_length
    while len(tokenizer(history_text).input_ids) < max_input_length and idx >= 0:
        user_message, chatbot_message = chat_history[idx]
        user_message = convert_message({"role": "user", "content": user_message})
        chatbot_message = convert_message({"role": "assistant", "content": chatbot_message})
        history_text = user_message + chatbot_message + history_text
        idx = idx - 1
    # if nothing was added, add <|assistant|> to start generation.
    if history_text == "":
        history_text = "<|assistant|>\n"
    return history_text

@torch.inference_mode()
def instruct(instruction, max_token_output=1024):
    input_text = instruction
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    input_ids = tokenizer(input_text, return_tensors='pt', truncation=False)
    input_ids["input_ids"] = input_ids["input_ids"].cuda()
    input_ids["attention_mask"] = input_ids["attention_mask"].cuda()
    generation_kwargs = dict(input_ids, streamer=streamer, max_new_tokens=max_token_output)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    return streamer


with gr.Blocks() as demo:
    # recreating the original qa demo in blocks
    with gr.Tab("QA Demo"):
        with gr.Row():
            instruction = gr.Textbox(label="Input")
            output = gr.Textbox(label="Output")
        greet_btn = gr.Button("Submit")
        def yield_instruct(instruction):
            # quick prompt hack:
            instruction = "<|user|>\n" + instruction + "\n<|assistant|>\n"
            output = ""
            for token in instruct(instruction):
                output += token
                yield output
        greet_btn.click(fn=yield_instruct, inputs=[instruction], outputs=output, api_name="greet")
    # chatbot-style model
    with gr.Tab("Chatbot"):
        chatbot = gr.Chatbot([], elem_id="chatbot")
        msg = gr.Textbox()
        clear = gr.Button("Clear")
        # fn to add user message to history
        def user(user_message, history):
            return "", history + [[user_message, None]]

    def bot(history):
        prompt = convert_history(history)
        streaming_out = instruct(prompt)
        history[-1][1] = ""
        for new_token in streaming_out:
            history[-1][1] += new_token
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue().launch(share=True)
