# Wordle Environment Setup - Fixed! ✅

## TL;DR
**Use Qwen3-0.6B with /no_think mode** for optimal base model training on Wordle environments.

## The Problem We Solved

Original issue: "🤡 After packing, there is not enough data to train"

Root causes:
1. **Tool schema was empty** - Environment tools had no parameter definitions
2. **Parameter name mismatch** - Model uses `"word"`, environment expects `"guess"`
3. **Base models over-think** - Qwen3 models spend all tokens in `<think>` tags
4. **Zero std filtering** - All samples got same reward → filtered out

## The Solution

### 1. Fixed Tool Schema
**File**: `open_instruct/environments/env_tool.py`

Added automatic parameter schema for Wordle environments:
```python
if "wordle" in env_name.lower():
    parameters = {
        "type": "object",
        "properties": {
            "word": {
                "type": "string",
                "description": "Your 5-letter guess for the Wordle game"
            }
        },
        "required": ["word"]
    }
```

### 2. Parameter Translation
**File**: `open_instruct/environments/prime_intellect.py`

Added translation for parameter name compatibility:
```python
# Translate "word" -> "guess" for compatibility
if "word" in action and "guess" not in action:
    action = {**action, "guess": action.pop("word")}
```

### 3. No-Think Mode
**File**: `no_think_system_prompt.txt`

Added `/no_think` instruction to prevent over-verbose thinking:
```
You are playing Wordle. Guess a 5-letter word. After each guess, you'll receive feedback: green (correct position), yellow (wrong position), gray (not in word). Use the wordle tool to submit guesses.

/no_think
```

### 4. Updated Training Script
**File**: `scripts/train/debug/envs/wordle_verifiers.sh`

Changed to use optimal configuration:
- Model: `Qwen/Qwen3-0.6B` (was `PrimeIntellect/Qwen3-1.7B-Wordle-SFT`)
- Added: `--system_prompt_override_file no_think_system_prompt.txt`
- Kept: `--response_length 1536`, `--filter_zero_std_samples false`

## Results Comparison

| Model | Config | Variance? | Mean Reward | Status |
|-------|--------|-----------|-------------|--------|
| Qwen3-1.7B | Normal | ❌ | 0.0 (all) | Too verbose |
| Qwen3-1.7B | /no_think + schema | ✅ | 0.80 | Works |
| **Qwen3-0.6B** | **/no_think + schema** | **✅** | **2.95** | **🥇 BEST** |
| Qwen2.5-0.5B | /no_think + schema | ❌ | 0.40 (all) | Too small |
| Qwen3-Wordle-SFT | Normal + schema | ✅ | ~4.0 | Pre-trained |

## Usage

### Two Implementations Available

#### 1. Prime Intellect Verifiers (Recommended)
```bash
# Fast, reliable, proven to work well
./scripts/train/debug/envs/wordle_verifiers.sh
```

#### 2. TextArena/OpenEnv
```bash
# Slower but uses standard OpenEnv protocol
./scripts/train/debug/envs/wordle_openenv.sh
```

**Note**: `wordle_verifiers.sh` is recommended for training as it's faster and more reliable. `wordle_openenv.sh` works but has slower rollouts due to websocket communication overhead.

### Alternative: Use Wordle-SFT Model
If you prefer the pre-trained model, edit `wordle_verifiers.sh`:
```bash
--model_name_or_path PrimeIntellect/Qwen3-1.7B-Wordle-SFT \
# Remove the --system_prompt_override_file line
```

## Key Files

1. **Training Script**: `scripts/train/debug/envs/wordle_verifiers.sh`
2. **System Prompt**: `no_think_system_prompt.txt`
3. **Tool Schema Fix**: `open_instruct/environments/env_tool.py` (line ~54)
4. **Parameter Translation**: `open_instruct/environments/prime_intellect.py` (line ~123)

## Testing

To verify the setup works:
```bash
# Check that games finish with varied rewards
grep "score_rollout complete" outputs/*/logs/*.log | tail -10

# Should see different reward values like: 0.1, 1.6, 3.8, 4.8
# NOT all the same value
```

## Notes

- The tool schema fix applies automatically to any environment with "wordle" in the name
- The parameter translation (`word` → `guess`) works for all Prime Intellect Wordle environments
- `/no_think` mode only works with Qwen3 models (not Qwen2.5)
- Smaller models (0.6B) surprisingly work better than larger ones (1.7B) with /no_think

## Credits

Fixed through systematic debugging:
1. Traced "not enough data" error to zero-std filtering
2. Discovered empty tool schema via prompt inspection
3. Found parameter name mismatch in environment logs
4. Tested multiple model sizes to find optimal configuration
