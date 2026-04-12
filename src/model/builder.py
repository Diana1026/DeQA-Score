#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

from src.model import *

def _load_non_lora_trainables(model, model_path):
    """Load extra trainable weights saved alongside LoRA adapters."""
    non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
    non_lora_trainables = None
    if os.path.exists(non_lora_path):
        non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
    else:
        # Try HF Hub if model_path is a repo_id.
        try:
            from huggingface_hub import hf_hub_download

            cache_file = hf_hub_download(
                repo_id=model_path,
                filename="non_lora_trainables.bin",
            )
            non_lora_trainables = torch.load(cache_file, map_location="cpu")
        except Exception:
            non_lora_trainables = None

    if non_lora_trainables is None:
        print("No non_lora_trainables.bin found; skip loading extra weights.")
        return

    # Normalize PEFT-prefixed names to base model names.
    normalized = {}
    for k, v in non_lora_trainables.items():
        nk = k
        if nk.startswith("base_model.model."):
            nk = nk[len("base_model.model.") :]
        normalized[nk] = v

    # If decomposition weights exist, make sure model has decomp modules before loading.
    has_decomp = any("decomp_" in k or "global_align_head" in k for k in normalized)
    if has_decomp and hasattr(model, "enable_decomp_embeddings"):
        num_decomp_tokens = None
        if "decomp_queries" in normalized:
            try:
                num_decomp_tokens = int(normalized["decomp_queries"].shape[0])
            except Exception:
                num_decomp_tokens = None
        model.enable_decomp_embeddings(num_decomp_tokens)
        print(
            f"Enabled decomposition modules before loading extra weights "
            f"(num_decomp_tokens={num_decomp_tokens})."
        )

    load_info = model.load_state_dict(normalized, strict=False)
    missing = len(load_info.missing_keys) if hasattr(load_info, "missing_keys") else 0
    unexpected = (
        len(load_info.unexpected_keys) if hasattr(load_info, "unexpected_keys") else 0
    )
    print(
        f"Loaded non-LoRA trainables: {len(normalized)} keys "
        f"(missing={missing}, unexpected={unexpected})"
    )


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    preprocessor_path=None,
):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if preprocessor_path is None:
        preprocessor_path = model_path

    if "deqa" in model_name.lower():
        # Load LLaVA model
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if "lora" in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(preprocessor_path, use_fast=False)
            print("Loading mPLUG-Owl2 from base model...")
            model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
            )
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, tokem_dim, device=model.device, dtype=model.dtype
                    )
                )
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, tokem_dim, device=model.device, dtype=model.dtype
                    )
                )

            print("Loading additional mPLUG-Owl2 weights...")
            _load_non_lora_trainables(model, model_path)

            from peft import PeftModel

            print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
            print("Model is loaded...")
        elif model_base is not None:
            # this may be mm projector only
            print("Loading mPLUG-Owl2 from base model...")
            tokenizer = AutoTokenizer.from_pretrained(preprocessor_path, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(preprocessor_path, use_fast=False)
            model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(preprocessor_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, **kwargs
            )
            print("Loading additional non-LoRA trainables (if available)...")
            _load_non_lora_trainables(model, model_path)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(preprocessor_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )

    # vision_tower = model.get_model().vision_model
    # print(vision_tower.device)
    # vision_tower.to(device=device, dtype=torch.float16)
    image_processor = CLIPImageProcessor.from_pretrained(preprocessor_path)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
