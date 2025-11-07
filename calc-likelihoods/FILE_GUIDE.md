# File Guide

This directory contains a complete toolkit for calculating and comparing response log likelihoods across multiple language models.

## 📄 Core Files

### `calculate_response_loglikelihoods.py`
**The main module** - Contains the core functionality.

**Key Functions:**
- `calculate_response_log_likelihood()` - Calculates log likelihood for a single prompt-response pair
- `evaluate_models_on_dataset()` - Evaluates multiple models on a full dataset

**Use this when:** You want to import the functions into your own scripts.

---

### `example_usage.py`
**Full-featured example** with data loading and visualization.

**Includes:**
- Examples for loading data from CSV, JSONL, and HuggingFace
- Complete visualization pipeline
- Report generation

**Use this when:** You want visualizations and detailed reports.

**Run it:**
```bash
python example_usage.py
```

---

### `quickstart.py`
**Minimal working example** - Get started in 30 seconds.

**What it does:**
- Uses built-in test data
- Evaluates with GPT-2 (smallest model)
- Shows basic results

**Use this when:** You want to test the setup quickly.

**Run it:**
```bash
python quickstart.py
```

---

### `config_template.py`
**Customizable template** - Copy and modify for your use case.

**Configure:**
- Data source (CSV, JSONL, HuggingFace, or list)
- Model list
- Column names
- Output options

**Use this when:** You have your own data and models ready.

**Run it:**
```bash
python config_template.py
```

---

## 📚 Documentation

### `README.md`
**Comprehensive documentation** covering:
- Installation instructions
- Usage examples
- Metric explanations
- Troubleshooting guide
- Performance tips

**Read this for:** Understanding how everything works.

---

### `requirements.txt`
**Python dependencies** needed to run the code.

**Install with:**
```bash
pip install -r requirements.txt --break-system-packages
```

---

### `FILE_GUIDE.md` (this file)
Quick reference for what each file does.

---

## 🚀 Quick Start Guide

### For First-Time Users:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt --break-system-packages
   ```

2. **Run the quickstart:**
   ```bash
   python quickstart.py
   ```

3. **Explore the example:**
   ```bash
   python example_usage.py
   ```

### For Your Own Data:

1. **Copy the template:**
   ```bash
   cp config_template.py my_config.py
   ```

2. **Edit configuration:**
   - Set your data source
   - Add your models
   - Adjust settings

3. **Run it:**
   ```bash
   python my_config.py
   ```

### For Custom Scripts:

```python
from calculate_response_loglikelihoods import evaluate_models_on_dataset
from datasets import Dataset

# Your code here
dataset = Dataset.from_list([...])
results = evaluate_models_on_dataset(dataset, ["gpt2", "gpt2-medium"])
```

---

## 📊 Output Files

When you run any of the scripts, you'll get:

### Always Generated:
- `log_likelihood_results.csv` - Detailed results for every sample and model

### Optional (with visualizations enabled):
- `log_likelihood_comparison.png` - Box plots and bar charts
- `perplexity_comparison.png` - Perplexity distributions
- `log_likelihood_heatmap.png` - Sample-wise comparison matrix
- `log_likelihood_trends.png` - Line plots across samples
- `comparison_report.txt` - Text summary and rankings

---

## 🎯 Which File Should I Use?

| Your Situation | Recommended File |
|----------------|------------------|
| Just want to test it works | `quickstart.py` |
| Need visualizations | `example_usage.py` |
| Have my own data ready | `config_template.py` |
| Building custom analysis | Import from `calculate_response_loglikelihoods.py` |
| Want to understand everything | `README.md` |
| Looking for specific function | This file! |

---

## 🔧 Common Modifications

### Change Models
Edit the `MODELS` list in any script:
```python
MODELS = [
    "gpt2",
    "gpt2-medium",
    "your-custom-model"
]
```

### Use Your Data
In `config_template.py`:
```python
USE_CSV = True
CSV_PATH = "/path/to/your/data.csv"
PROMPT_COLUMN = "your_prompt_column"
RESPONSE_COLUMN = "your_response_column"
```

### Disable Visualizations
In scripts that support it:
```python
CREATE_VISUALIZATIONS = False
```

### Use CPU Instead of GPU
```python
DEVICE = "cpu"
```

---

## 📈 Example Workflows

### Workflow 1: Quick Test
```bash
python quickstart.py
# Check quickstart_results.csv
```

### Workflow 2: Full Analysis
```bash
# 1. Prepare data.csv with 'prompt' and 'response' columns
# 2. Edit config_template.py:
#    - USE_CSV = True
#    - CSV_PATH = "data.csv"
#    - MODELS = ["model1", "model2", "model3"]
# 3. Run
python config_template.py
# 4. Review outputs in /mnt/user-data/outputs/
```

### Workflow 3: Custom Script
```python
from calculate_response_loglikelihoods import evaluate_models_on_dataset
from datasets import Dataset

# Your custom data loading
data = load_my_special_data()
dataset = Dataset.from_list(data)

# Your custom model list
models = get_my_model_checkpoints()

# Run evaluation
results = evaluate_models_on_dataset(dataset, models)

# Your custom analysis
analyze_results(results)
```

---

## ❓ Getting Help

1. **Read README.md** - Comprehensive guide
2. **Check examples** - Look at how others do it
3. **Error messages** - Often point to the issue
4. **Test with quickstart** - Isolate the problem

---

## 📝 Notes

- All scripts assume you have PyTorch and transformers installed
- First run will download models (can be large!)
- GPU recommended for large models
- Start with small test datasets
- Results are saved to `/mnt/user-data/outputs/` by default

---

## 🎓 Learning Path

1. **Beginner**: Run `quickstart.py` → Read `README.md`
2. **Intermediate**: Modify `config_template.py` with your data
3. **Advanced**: Import functions and build custom analysis
4. **Expert**: Extend `calculate_response_loglikelihoods.py` for your needs

---

Happy analyzing! 🚀
