"""Inference generator with medical safety guardrails and schema validation."""

import logging
from typing import Any, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.dataset.prompts import BIOEVIDENCE_SYSTEM_PROMPT, build_user_prompt
from src.inference.schema_parser import parse_and_validate_output

logger = logging.getLogger(__name__)

MANDATORY_MEDICAL_DISCLAIMER = (
    "DISCLAIMER: This system is intended for biomedical research and educational use. "
    "It is not a substitute for professional medical advice, diagnosis, treatment, or clinical decision-making."
)


class BioEvidenceGenerator:
    """Inference engine for BioEvidence-LLM."""

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.tokenizer = None
        self.model = None
        self.load_in_4bit = load_in_4bit and (device == "cuda")

    def load_model(self):
        """Load tokenizer, base model (4-bit QLoRA if specified), and adapter."""
        if self.model is not None:
            return

        logger.info("Loading tokenizer from %s", self.base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        logger.info("Loading base model: %s", self.base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )

        if self.adapter_path:
            logger.info("Loading LoRA adapter from %s", self.adapter_path)
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.model.eval()

    def generate(
        self,
        question: Optional[str],
        context: str,
        task: str = "evidence_qa",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate evidence-grounded answer given question and context."""
        self.load_model()

        user_content = build_user_prompt(question, context, task)
        messages = [
            {"role": "system", "content": BIOEVIDENCE_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_content.strip()},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Slice generation past the prompt tokens
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        raw_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        parsed_output = parse_and_validate_output(raw_response)

        return {
            "disclaimer": MANDATORY_MEDICAL_DISCLAIMER,
            "raw_output": raw_response,
            "structured": parsed_output.model_dump(),
        }
