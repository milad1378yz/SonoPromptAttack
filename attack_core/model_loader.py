import os
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)


def _resolve_vlm_device_map():
    raw = str(os.getenv("VLM_DEVICE_MAP", "auto")).strip()
    if not raw or raw.lower() == "auto":
        return "auto"
    if raw.isdigit():
        return {"": int(raw)}
    lowered = raw.lower()
    if lowered.startswith("cuda:") or lowered == "cpu":
        return {"": raw}
    return raw


def _resolve_local_files_only():
    raw = str(os.getenv("HF_LOCAL_FILES_ONLY", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _inference_dtype(*, use_cuda: bool):
    if not use_cuda:
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_vlm(model_id: str):
    device_map = _resolve_vlm_device_map()
    targets_cpu = isinstance(device_map, dict) and device_map.get("") == "cpu"
    dtype = _inference_dtype(use_cuda=torch.cuda.is_available() and not targets_cpu)
    local_files_only = _resolve_local_files_only()

    # Use HF_HOME as a cache location, not as an implicit offline-mode switch.
    cache_dir = os.getenv("HF_HOME")
    if cache_dir:
        print(f"Using local cache directory: {cache_dir}")
    if local_files_only:
        print("HF_LOCAL_FILES_ONLY is enabled; loading VLM in offline-only mode.")

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    if "llava-med" in model_id or "llava" in model_id:
        print("Detected LLaVA-style model, using LlavaForConditionalGeneration.")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    else:

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    # Show the model devices in short form (only the devices used and not the layers on them (a list))
    model_devices = set()
    for _, param in model.named_parameters():
        model_devices.add(str(param.device))
    model_devices = ", ".join(sorted(model_devices))
    print(f"Loaded VLM model '{model_id}' on devices: {model_devices} (device_map={device_map})")
    return model, processor


def _should_use_4bit(model_id: str, quantization: str) -> bool:
    if quantization == "4bit":
        return True
    if quantization == "fp16":
        return False

    lowered = model_id.lower()
    heavy_tokens = ("30b", "32b", "34b", "35b", "36b", "65b", "70b", "72b", "110b", "120b")
    return any(tok in lowered for tok in heavy_tokens)


def _load_llm_4bit(model_id: str, dtype):
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit LLM loading requires a CUDA-capable GPU.")
    torch.cuda.empty_cache()
    bnb_conf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )
    offload_dir = os.getenv("LLM_OFFLOAD_DIR", "cache/llm_offload")
    Path(offload_dir).mkdir(parents=True, exist_ok=True)
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=bnb_conf,
        offload_folder=offload_dir,
    )


def load_llm(model_id: str, quantization: str = "auto"):
    dtype = _inference_dtype(use_cuda=torch.cuda.is_available())
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    common_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    use_4bit = _should_use_4bit(model_id, quantization)
    if use_4bit:
        print(f"Loading LLM '{model_id}' in 4-bit mode (quantization={quantization}).")
        mdl = _load_llm_4bit(model_id, dtype)
    else:
        try:
            mdl = AutoModelForCausalLM.from_pretrained(model_id, **common_kwargs)
        except torch.cuda.OutOfMemoryError as exc:
            print(
                f"LLM standard load OOM ({exc}); retrying with 4-bit quantization. "
                "Use --llm-quantization 4bit to avoid this fallback."
            )
            mdl = _load_llm_4bit(model_id, dtype)

    # Show the model devices in short form (only the devices used and not the layers on them (a list))
    model_devices = set()
    for _, param in mdl.named_parameters():
        model_devices.add(str(param.device))
    model_devices = ", ".join(sorted(model_devices))
    print(f"Loaded LLM model '{model_id}' on devices: {model_devices}")
    return mdl, tok
