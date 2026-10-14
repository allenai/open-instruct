import json

from scripts.miles import probe_opd_teacher_eos


def test_probe_scores_same_prefix_without_generation(monkeypatch, tmp_path):
    panel, output = tmp_path / "panel.json", tmp_path / "result.json"
    panel.write_text(
        json.dumps(
            [
                {
                    "input_ids": [10, 20, 248046],
                    "saved_teacher_sampled_logprob": -0.25,
                    "saved_student_sampled_logprob": -0.1,
                }
            ]
        )
    )
    calls = []

    def request(url, payload=None):
        calls.append((url, payload))
        if payload is None:
            return {"model_path": "frozen-teacher"}
        return {
            "meta_info": {
                "input_token_logprobs": [[-0.25, 248046, None]],
                "input_token_ids_logprobs": [[[-12.0, 248044, None], [-0.25, 248046, None]]],
            }
        }

    monkeypatch.setattr(probe_opd_teacher_eos, "request", request)
    monkeypatch.setattr("sys.argv", ["probe", str(panel), "--url", "http://localhost:1234", "--output", str(output)])
    probe_opd_teacher_eos.main()
    assert calls[1][1]["input_ids"] == [10, 20, 248046]
    assert calls[1][1]["sampling_params"]["max_new_tokens"] == 0
    assert calls[1][1]["logprob_start_len"] == 1
    result = json.loads(output.read_text())
    assert result["rows"][0]["im_end_rescore_minus_saved"] == 0
    assert result["rows"][0]["teacher_endoftext_logprob"] == -12
