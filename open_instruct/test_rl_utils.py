import unittest

import numpy as np
import torch
import transformers

import open_instruct.rl_utils


def get_test_data():
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    prompts = [
        "User: Hello, how are you?\nAssistant: <think>",
        "User: What is the capital of France?\nAssistant: <think>",
        "User: What is the capital of Germany?\nAssistant: <think>",
    ]
    outputs = ["I'm good, thank you!", "Paris", "Berlin"]
    queries = [tokenizer.encode(prompt) for prompt in prompts]
    responses = [tokenizer.encode(response) for response in outputs]
    prompt_max_len = 20
    generate_max_len = 20
    assert all(len(query) <= prompt_max_len for query in queries)
    assert all(len(response) <= generate_max_len for response in responses)
    return queries, responses, tokenizer.pad_token_id


class TestRLUtils(unittest.TestCase):
    def test_pack_sequences(self):
        queries, responses, pad_token_id = get_test_data()
        pack_length = 40
        masks = [[1] * len(response) for response in responses]
        vllm_logprobs = [[0.0] * len(response) for response in responses]
        with open_instruct.rl_utils.Timer("pack_sequences"):
            packed_sequences = open_instruct.rl_utils.pack_sequences(
                queries=queries,
                responses=responses,
                masks=masks,
                pack_length=pack_length,
                pad_token_id=pad_token_id,
                vllm_logprobs=vllm_logprobs,
            )

        offset = 0
        np.testing.assert_allclose(
            packed_sequences.query_responses[0][offset : offset + len(queries[0]) + len(responses[0])],
            np.array(sum([queries[0], responses[0]], [])),
        )
        offset += len(queries[0]) + len(responses[0])
        np.testing.assert_allclose(
            packed_sequences.query_responses[0][offset : offset + len(queries[1]) + len(responses[1])],
            np.array(sum([queries[1], responses[1]], [])),
        )
        offset = 0
        np.testing.assert_allclose(
            packed_sequences.query_responses[1][offset : offset + len(queries[2]) + len(responses[2])],
            np.array(sum([queries[2], responses[2]], [])),
        )

    def test_calculate_advantages_packed(self):
        gamma = 1
        lam = 1
        values = np.array(
            [
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            ]
        )
        rewards = np.zeros((3, 13))
        rewards[0, 5] = 10
        rewards[1, 6] = 20
        rewards[2, 12] = 30
        adv, ret = open_instruct.rl_utils.calculate_advantages(values, rewards, gamma, lam)

        packed_response_masks = np.zeros((1, 42), dtype=int)
        packed_response_masks[0, 5:11] = 1
        packed_response_masks[0, 15:22] = 1
        packed_response_masks[0, 29:42] = 1

        packed_values = np.full((1, 41), 5)
        packed_values[0, 4:10] = 1
        packed_values[0, 14:21] = 2
        packed_values[0, 28:41] = 3

        packed_values_masked = np.where(packed_response_masks[:, 1:] == 0, 0, packed_values)

        packed_rewards = np.zeros((1, 42))
        packed_rewards[0, 10] = 10
        packed_rewards[0, 21] = 20
        packed_rewards[0, 41] = 30

        packed_dones = np.zeros((1, 42))
        packed_dones[0, 10] = 1
        packed_dones[0, 21] = 2
        packed_dones[0, 41] = 3

        packed_adv, packed_ret = open_instruct.rl_utils.calculate_advantages_packed(
            packed_values_masked, packed_rewards[:, 1:], gamma, lam, packed_dones[:, 1:], packed_response_masks[:, 1:]
        )

        packed_values = np.full((1, 37), -1)
        packed_values[0, 4:9] = 1
        packed_values[0, 12:18] = 2
        packed_values[0, 24:36] = 3
        packed_values[0, 36] = 50256

        packed_rewards = np.zeros((1, 38))
        packed_rewards[0, 8] = 10
        packed_rewards[0, 17] = 20
        packed_rewards[0, 35] = 30
        packed_rewards[0, 37] = 50256

        packed_dones = np.zeros((1, 37))
        packed_dones[0, 8] = 1
        packed_dones[0, 17] = 1
        packed_dones[0, 35] = 1
        packed_dones[0, 36] = 50256

        packed_response_masks = np.zeros((1, 37), dtype=int)
        packed_response_masks[0, 4:9] = 1
        packed_response_masks[0, 12:18] = 1
        packed_response_masks[0, 24:36] = 1
        packed_response_masks[0, 36] = 50256
        packed_adv, packed_ret = open_instruct.rl_utils.calculate_advantages_packed(
            packed_values, packed_rewards, gamma, lam, packed_dones, packed_response_masks
        )

        np.testing.assert_allclose(adv[0, :5], packed_adv[0, 4:9])
        np.testing.assert_allclose(ret[1, :6], packed_ret[0, 12:18])
        np.testing.assert_allclose(adv[2, :12], packed_adv[0, 24:36])

    def test_pack_sequences_logits(self):
        transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")

        model = transformers.AutoModelForCausalLM.from_pretrained(
            "EleutherAI/pythia-14m", attn_implementation="eager", torch_dtype=torch.bfloat16
        )
        value_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            "EleutherAI/pythia-14m", num_labels=1, torch_dtype=torch.bfloat16
        )
        value_model.train()
        value_head = value_model.score
        config = value_model.config
        torch.manual_seed(2)
        torch.nn.init.normal_(value_head.weight, std=1 / (config.hidden_size + 1))
        print(value_head.weight)
        value_model.save_pretrained("pythia-14m-critic")
        queries, responses, pad_token_id = get_test_data()
        query_responses = [q + r for q, r in zip(queries, responses)]
        pack_length = 40
        masks = [[1] * len(response) for response in responses]
        vllm_logprobs = [[0.0] * len(response) for response in responses]
        with open_instruct.rl_utils.Timer("pack_sequences"):
            packed_sequences = open_instruct.rl_utils.pack_sequences(
                queries=queries,
                responses=responses,
                masks=masks,
                pack_length=pack_length,
                pad_token_id=pad_token_id,
                vllm_logprobs=vllm_logprobs,
            )
        lm_backbone = getattr(value_model, value_model.base_model_prefix)
        torch.manual_seed(2)
        v = lm_backbone(
            input_ids=packed_sequences.query_responses[0].unsqueeze(0),
            attention_mask=packed_sequences.attention_masks[0].unsqueeze(0).clamp(0, 1),
            position_ids=packed_sequences.position_ids[0].unsqueeze(0),
        )
        values = value_head(v.last_hidden_state).squeeze(-1)
        values = torch.where(packed_sequences.response_masks[0].unsqueeze(0) == 0, 0, values)
        s = model.forward(
            input_ids=packed_sequences.query_responses[0].unsqueeze(0)[:, :-1],
            attention_mask=packed_sequences.attention_masks[0].unsqueeze(0)[:, :-1],
            position_ids=packed_sequences.position_ids[0].unsqueeze(0)[:, :-1],
        )
        all_logprobs = torch.nn.functional.log_softmax(s.logits, dim=-1)
        logprobs = torch.gather(
            all_logprobs, 2, packed_sequences.query_responses[0].unsqueeze(0).unsqueeze(-1)[:, 1:]
        ).squeeze(-1)
        logprobs = torch.where(packed_sequences.response_masks[0].unsqueeze(0)[:, 1:] == 0, 1.0, logprobs)

        logprobs = []
        for i in range(len(query_responses)):
            query_response = query_responses[i]
            query = queries[i]
            s2 = model.forward(
                input_ids=torch.tensor([query_response])[:, :-1],
                attention_mask=torch.tensor([query_response])[:, :-1] != pad_token_id,
            )
            all_logprobs = torch.nn.functional.log_softmax(s2.logits, dim=-1)
            logprob = torch.gather(all_logprobs, 2, torch.tensor([query_response]).unsqueeze(-1)[:, 1:]).squeeze(-1)
            logprobs.append(logprob[:, len(query) - 1 :])

        rewards = np.zeros_like(values.detach().float().numpy())
        rewards[:, 21] = 0.1
        rewards[:, -1] = 1
        adv, ret = open_instruct.rl_utils.calculate_advantages_packed(
            values=values.detach().float().numpy(),
            rewards=rewards,
            gamma=1.0,
            lam=1.0,
            dones=packed_sequences.dones[0].unsqueeze(0).numpy(),
            response_masks=packed_sequences.response_masks[0].unsqueeze(0).numpy(),
        )


if __name__ == "__main__":
    unittest.main()
