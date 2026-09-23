# GGUF Export & Local Offline Inference Guide

> **Target Quantizations:** Q4_K_M, Q8_0  
> **Target Execution Engines:** `llama.cpp` and Ollama  
> **Model Target:** `BioEvidence-LLM` (merged Qwen2.5-1.5B adapter)

---

## 1. Merging LoRA Adapter with Base Model

Before GGUF quantization, the trained LoRA adapter weights must be merged into the unquantized base model weights:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_dir = "models/adapters/bioevidence-lora-best"
merged_output_dir = "models/base/bioevidence-merged-fp16"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
merged_model = model.merge_and_unload()

merged_model.save_pretrained(merged_output_dir)
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
tokenizer.save_pretrained(merged_output_dir)
```

---

## 2. GGUF Conversion via `llama.cpp`

1. Clone and compile `llama.cpp`:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   cmake -B build
   cmake --build build --config Release
   ```

2. Convert Hugging Face weights to GGUF (FP16):
   ```bash
   python convert_hf_to_gguf.py models/base/bioevidence-merged-fp16/ --outfile models/gguf/bioevidence-qwen2.5-1.5b-fp16.gguf
   ```

3. Quantize to 4-bit Medium (`Q4_K_M`):
   ```bash
   ./build/bin/llama-quantize models/gguf/bioevidence-qwen2.5-1.5b-fp16.gguf models/gguf/bioevidence-qwen2.5-1.5b-q4_k_m.gguf Q4_K_M
   ```

---

## 3. Local Execution via Ollama

Create a `Modelfile`:
```dockerfile
FROM ./models/gguf/bioevidence-qwen2.5-1.5b-q4_k_m.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
SYSTEM """You are BioEvidence-LLM. Always respond strictly in valid JSON with decision, answer, evidence, uncertainty, and limitations."""
```

Build and run:
```bash
ollama create bioevidence -f Modelfile
ollama run bioevidence "Does intervention X reduce mortality in condition Y?"
```
