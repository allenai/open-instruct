"""Repro script for olmo3_2_hybrid model_type not recognized by new transformers fork.

The new transformers fork (olmo-3.5-hybrid-clean branch) renamed the model type
from "olmo3_2_hybrid" to "olmo_hybrid", but existing HF checkpoints still have
"model_type": "olmo3_2_hybrid" in their config.json, causing AutoConfig to fail.
"""

import json
import tempfile

from transformers import AutoConfig


def test_local_config():
    """Create a minimal config.json with the old model_type and try to load it."""
    minimal_config = {
        "model_type": "olmo3_2_hybrid",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 256,
        "vocab_size": 32000,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = f"{tmpdir}/config.json"
        with open(config_path, "w") as f:
            json.dump(minimal_config, f)

        print(f"Created minimal config at {config_path}")
        print(f"Config contents: {json.dumps(minimal_config, indent=2)}")
        print()

        try:
            config = AutoConfig.from_pretrained(tmpdir)
            print(f"SUCCESS: AutoConfig loaded. model_type={config.model_type}")
        except KeyError as e:
            print(f"FAILED (KeyError): {e}")
            print(
                "The 'olmo3_2_hybrid' model_type is not in CONFIG_MAPPING_NAMES."
            )
        except Exception as e:
            print(f"FAILED ({type(e).__name__}): {e}")


def test_hf_checkpoint():
    """Try loading the real HF checkpoint."""
    model_name = "allenai/Olmo-Hybrid-Instruct-DPO-7B"
    print(f"Attempting AutoConfig.from_pretrained('{model_name}')...")

    try:
        config = AutoConfig.from_pretrained(model_name)
        print(f"SUCCESS: AutoConfig loaded. model_type={config.model_type}")
    except KeyError as e:
        print(f"FAILED (KeyError): {e}")
    except OSError as e:
        print(f"SKIPPED (auth/network issue): {e}")
    except Exception as e:
        print(f"FAILED ({type(e).__name__}): {e}")


def main():
    print("=" * 60)
    print("Repro: olmo3_2_hybrid model_type not recognized")
    print("=" * 60)
    print()

    print("--- Test 1: Local minimal config with old model_type ---")
    test_local_config()
    print()

    print("--- Test 2: Real HF checkpoint ---")
    test_hf_checkpoint()
    print()

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
