# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Demo: SAM 3 Agent with Ollama Cloud API (VLM backend).
# Run from project root: python demo_sam3_agent.py
# Set your API key and model name below; behavior matches examples/sam3_agent.ipynb.

import base64
import io
import json
import os
from functools import partial

import numpy as np
import pycocotools.mask as mask_utils
import torch
from openai import OpenAI
from PIL import Image, ImageOps

import sam3
from sam3 import build_sam3_image_model
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference
from sam3.model.sam3_image_processor import Sam3Processor

# --- Configure these (Ollama Cloud API key from https://ollama.com/settings/keys) ---
OLLAMA_API_KEY = "391877a2fd204a809e36c496a1ff64c1.xxpLOf9O7Mf3rFNhOAjFVWo6"
OLLAMA_MODEL = "qwen3.5:cloud"
IMAGE_PATH = "demo/img_5.png"
PROMPT = "the blue chair"
# -----------------------------------------------------------------------------------

OLLAMA_BASE_URL = "https://ollama.com/v1"
OUTPUT_DIR = "agent_output"
RESULTS_DIR = "demo/results"
MAX_TOKENS = 4096
# Resize images with longer side > this before sending to Ollama to avoid 500 (payload too large)
MAX_IMAGE_DIMENSION = 1024


def save_clean_results(pred_json_path: str, results_dir: str = RESULTS_DIR):
    """
    Load agent pred JSON, decode RLE masks, and save:
    - {name}_rgb.png: object on black background
    - {name}_mask.png: binary mask (255=object, 0=background)
    """
    if not os.path.exists(pred_json_path):
        print(f"Pred JSON not found: {pred_json_path}")
        return
    with open(pred_json_path) as f:
        data = json.load(f)
    orig_h = int(data["orig_img_h"])
    orig_w = int(data["orig_img_w"])
    img_path = data.get("original_image_path") or data.get("image_path")
    pred_masks = data.get("pred_masks", [])
    if not pred_masks:
        print("No masks in pred JSON, skipping save_clean_results.")
        return
    # Merge all RLE masks into one binary mask
    combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for rle_str in pred_masks:
        rle = {"size": (orig_h, orig_w), "counts": rle_str}
        binary = mask_utils.decode(rle)
        combined = np.maximum(combined, binary)
    # Load original image with EXIF correction
    pil_img = Image.open(img_path).convert("RGB")
    pil_img = ImageOps.exif_transpose(pil_img)
    rgb = np.array(pil_img)
    if rgb.shape[:2] != (orig_h, orig_w):
        print(
            f"Warning: image shape {rgb.shape[:2]} != mask shape ({orig_h}, {orig_w}), skipping."
        )
        return
    # Object on black background
    rgb_on_black = rgb.copy()
    rgb_on_black[combined == 0] = 0
    # Binary mask (255 = object, 0 = background)
    mask_uint8 = (combined > 0).astype(np.uint8) * 255
    os.makedirs(results_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pred_json_path))[0].replace(
        "_pred", ""
    )
    rgb_path = os.path.join(results_dir, f"{base_name}_rgb.png")
    mask_path = os.path.join(results_dir, f"{base_name}_mask.png")
    Image.fromarray(rgb_on_black).save(rgb_path)
    Image.fromarray(mask_uint8).save(mask_path)
    print(f"Results saved: {rgb_path}, {mask_path}")


def _image_to_base64_for_ollama(image_path: str):
    """
    Load image, optionally resize if too large, return (base64_str, mime_type).
    Reduces payload size so Ollama Cloud does not return 500 on multi-image requests.
    """
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime_type = mime_types.get(ext, "image/jpeg")
    path_escaped = image_path.replace("?", "%3F")
    img = Image.open(path_escaped).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def send_generate_request_ollama(
    messages,
    server_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    api_key=OLLAMA_API_KEY,
    max_tokens=MAX_TOKENS,
):
    """
    Ollama Cloud–compatible variant of send_generate_request.
    Uses OpenAI client against https://ollama.com/v1 with max_tokens (no n=1, no detail=high).
    """
    processed_messages = []
    for message in messages:
        processed_message = message.copy()
        if message["role"] == "user" and "content" in message:
            processed_content = []
            for c in message["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    image_path = c["image"]
                    print("image_path", image_path)
                    try:
                        base64_image, mime_type = _image_to_base64_for_ollama(
                            image_path
                        )
                        processed_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                },
                            }
                        )
                    except FileNotFoundError:
                        print(f"Warning: Image file not found: {image_path}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing image {image_path}: {e}")
                        continue
                else:
                    processed_content.append(c)
            processed_message["content"] = processed_content
        processed_messages.append(processed_message)

    client = OpenAI(api_key=api_key, base_url=server_url)
    try:
        print(f"🔍 Calling model {model}...")
        response = client.chat.completions.create(
            model=model,
            messages=processed_messages,
            max_tokens=max_tokens,
        )
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        print(f"Unexpected response format: {response}")
        return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def main():
    # Env setup (same as notebook)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()

    SAM3_ROOT = os.path.dirname(os.path.abspath(__file__))
    os.chdir(SAM3_ROOT)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    _ = os.system("nvidia-smi")

    # Build SAM3 model
    sam3_root = os.path.dirname(sam3.__file__)
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=0.5)

    # LLM config for Ollama Cloud
    llm_config = {
        "name": "ollama_cloud",
        "model": OLLAMA_MODEL,
        "api_key": OLLAMA_API_KEY,
        "base_url": OLLAMA_BASE_URL,
    }

    image_path = os.path.abspath(IMAGE_PATH)
    send_generate_request = partial(
        send_generate_request_ollama,
        server_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        api_key=OLLAMA_API_KEY,
        max_tokens=MAX_TOKENS,
    )
    call_sam_service = partial(
        call_sam_service_orig,
        sam3_processor=processor,
    )

    output_image_path = run_single_image_inference(
        image_path,
        PROMPT,
        llm_config,
        send_generate_request,
        call_sam_service,
        debug=True,
        output_dir=OUTPUT_DIR,
    )
    if output_image_path is not None:
        print(f"Output image saved: {output_image_path}")
        pred_json_path = output_image_path.replace("_pred.png", "_pred.json")
        save_clean_results(pred_json_path, results_dir=RESULTS_DIR)


if __name__ == "__main__":
    main()
