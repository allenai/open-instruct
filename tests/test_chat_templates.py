import pytest

from open_instruct.dataset_transformation import (
    sft_span_seach_mask_out,
    INPUT_IDS_KEY, LABELS_KEY
)
from transformers import AutoTokenizer
from typing import List

MSG_SYSTEM = { 'role': 'system', 'content': 'This is an system.' }
MSG_USER_1 = { 'role': 'user', 'content': 'This is an instruction.' }
MSG_USER_2 = { 'role': 'user', 'content': 'This is another instruction.' }
MSG_ASST_1 = { 'role': 'assistant', 'content': 'This is a response.' }
MSG_ASST_2 = { 'role': 'assistant', 'content': 'This is another response.' }
MSG_ASST_THINK_1 = { 'role': 'assistant', 'content': '<think> I am thinking. </think> This is a response.' }
MSG_ASST_THINK_2 = { 'role': 'assistant', 'content': '<think> I am thinking again. </think> This is another response.' }
MSG_TOOL = { 'role': 'tool', 'content': 'this is a response from tool' }

EXAMPLES = {
    'single-turn': {
        'messages': [ MSG_USER_1, MSG_ASST_1 ]
    },
    'multi-turn': {
        'messages': [ MSG_SYSTEM, MSG_USER_1, MSG_ASST_1, MSG_USER_2, MSG_ASST_2 ]
    },
    'multi-turn-with-think': {
        'messages': [ MSG_SYSTEM, MSG_USER_1, MSG_ASST_THINK_1, MSG_USER_2, MSG_ASST_THINK_2 ]
    },
    'multi-turn-with-think-tools': {
        'messages': [ MSG_SYSTEM, MSG_USER_1, MSG_ASST_THINK_1, MSG_TOOL, MSG_USER_2, MSG_ASST_THINK_2 ],
        'tools': [
            {'type': 'function'},
        ]
    },
}


CHAT_TEMPLATE_EXAMPLES = {
    'thinking': '{%- if tools %}\n    {{- \'<|im_start|>system\\n\' }}\n    {%- if messages[0].role == \'system\' %}\n        {{- messages[0].content + \'\\n\\n\' }}\n    {%- endif %}\n    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n    {%- for tool in tools %}\n        {{- "\\n" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}\n    {%- if messages[0].role == \'system\' %}\n        {{- \'<|im_start|>system\\n\' + messages[0].content + \'<|im_end|>\\n\' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith(\'<tool_response>\') and message.content.endswith(\'</tool_response>\')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = \'\' %}\n    {%- endif %}\n    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}\n        {{- \'<|im_start|>\' + message.role + \'\\n\' + content + \'<|im_end|>\' + \'\\n\' }}\n    {%- elif message.role == "assistant" %}\n        {%- set reasoning_content = \'\' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if \'</think>\' in content %}\n                {%- set reasoning_content = content.split(\'</think>\')[0].rstrip(\'\\n\').split(\'<think>\')[-1].lstrip(\'\\n\') %}\n                {%- set content = content.split(\'</think>\')[-1].lstrip(\'\\n\') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- \'<|im_start|>\' + message.role + \'\\n<think>\\n\' + reasoning_content.strip(\'\\n\') + \'\\n</think>\\n\\n\' + content.lstrip(\'\\n\') }}\n            {%- else %}\n                {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- \'\\n\' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- \'<tool_call>\\n{"name": "\' }}\n                {{- tool_call.name }}\n                {{- \'", "arguments": \' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- \'}\\n</tool_call>\' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- \'<|im_end|>\\n\' }}\n    {%- elif message.role == "tool" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}\n            {{- \'<|im_start|>user\' }}\n        {%- endif %}\n        {{- \'\\n<tool_response>\\n\' }}\n        {{- content }}\n        {{- \'\\n</tool_response>\' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\n            {{- \'<|im_end|>\\n\' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|im_start|>assistant\\n\' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- \'<think>\\n\\n</think>\\n\\n\' }}\n    {%- endif %}\n{%- endif %}'
}

ANSWERS = {
    ('Qwen/Qwen3-8B', 'thinking'): {
        'single-turn': [[0, 12]],
        'multi-turn': [[0, 22], [30, 42]],
        'multi-turn-with-think': [[0, 22], [30, 42]],
        'multi-turn-with-think-tools': [[0, 105], [113, 140]],
    }
}

TAGS = {
    'Qwen/Qwen3-8B': {
        'asst_tag': '<|im_start|>assistant',
        'end_tag': '<|im_end|>'
    }
}

MAX_SEQ_LENGTH = 1024

@pytest.mark.parametrize(
    "tokenizer_name,chat_template_name,example_name", 
    [
        ('Qwen/Qwen3-8B', 'thinking', 'single-turn'),
        ('Qwen/Qwen3-8B', 'thinking', 'multi-turn'),
        ('Qwen/Qwen3-8B', 'thinking', 'multi-turn-with-think'),
        ('Qwen/Qwen3-8B', 'thinking', 'multi-turn-with-think-tools'),
    ]
)
def test_masking_strategy_span_search(
    tokenizer_name: str,
    chat_template_name: str,
    example_name: List,
    ignore_label: int = -100,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # patch the chat template
    tokenizer.chat_template = CHAT_TEMPLATE_EXAMPLES[chat_template_name]

    tags = TAGS[tokenizer_name]

    output = sft_span_seach_mask_out(
        EXAMPLES[example_name],
        tokenizer,
        MAX_SEQ_LENGTH,
        asst_tag=tags['asst_tag'],
        end_tag=tags['end_tag'],
        ignore_label=ignore_label,
    )
    input_ids = output[INPUT_IDS_KEY]
    labels = output[LABELS_KEY]


    spans = []
    start = False
    for i in range(len(labels)):
        if labels[i] == ignore_label and not start:
            spans.append([i])
            start = True
        elif labels[i] != ignore_label and start:
            spans[-1].append(i)
            start = False

    for s, e in spans:
        labels[s:e] = -ignore_label


    print('*' * 10, 'original', '*' * 10)
    print (tokenizer.decode(input_ids[0]))
    print('*' * 10, 'masked', '*' * 10)
    print (tokenizer.decode(labels))
    print('*' * 10, 'spans', '*' * 10)
    print (spans)

    assert ANSWERS[(
        tokenizer_name, chat_template_name
    )][example_name] == spans

    

@pytest.mark.parametrize(
    "tokenizer_name,chat_template_name", 
    [
        ('Qwen/Qwen3-8B', 'thinking')
    ]
)
def test_masking_strategy_raises(
    tokenizer_name: str,
    chat_template_name: str,
):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # patch the chat template
    tokenizer.chat_template = CHAT_TEMPLATE_EXAMPLES[chat_template_name]

    # this simulates the case where a user set the wrong
    # tags in the function that does not match the chat-template
    with pytest.raises(ValueError):
        sft_span_seach_mask_out(
            {
                'messages': [{
                    'role': 'user', 'content': ''
                }]
            },
            tokenizer,
            MAX_SEQ_LENGTH,
            asst_tag='supersupersuperlongtag',
            end_tag='supersupersuperlongtag',
        )

