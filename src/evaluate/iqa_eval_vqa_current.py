import argparse
import json
import math
import os
from io import BytesIO

import requests
import torch
from PIL import Image
from tqdm import tqdm

from src.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from src.conversation import conv_templates
from src.mm_utils import get_model_name_from_path, tokenizer_image_token
from src.model.builder import load_pretrained_model


def disable_torch_init():
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def build_question(text_prompt):
    return f'Does this image match the text prompt "{text_prompt}"? Please answer yes or no.'


def build_vqa_prompt(conv_mode, question):
    conv = conv_templates[conv_mode].copy()
    inp = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def resolve_yes_no_ids(model, tokenizer):
    yes_id = getattr(model.config, "align_yes_token_id", None)
    no_id = getattr(model.config, "align_no_token_id", None)
    if yes_id is not None and no_id is not None:
        return int(yes_id), int(no_id)

    yes_ids = tokenizer("Yes", add_special_tokens=False)["input_ids"]
    no_ids = tokenizer("No", add_special_tokens=False)["input_ids"]
    if len(yes_ids) == 0 or len(no_ids) == 0:
        raise ValueError("Cannot tokenize 'Yes'/'No' to valid token ids.")
    return int(yes_ids[0]), int(no_ids[0])


@torch.inference_mode()
def infer_one(
    model,
    tokenizer,
    image_tensor,
    prompt,
    yes_id,
    no_id,
    device,
    use_align_residual=True,
    fuse_alpha=1.0,
    keep_details=False,
):
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=image_tensor.to(device, non_blocking=True),
        return_dict=True,
        use_alignment_branch=True,
    )

    next_logits = outputs.logits[:, -1, :]  # next token logits at answer position
    yes_logit = next_logits[:, yes_id]
    no_logit = next_logits[:, no_id]
    vqa_margin = yes_logit - no_logit
    vqa_p_yes = torch.sigmoid(vqa_margin)

    # Keep main branch score for debugging/comparison.
    vqa_pred_score_1to5 = 1.0 + 4.0 * vqa_p_yes
    margin = vqa_margin
    p_yes = torch.sigmoid(margin)

    out = {
        "pred_score_1to5": float((1.0 + 4.0 * p_yes).item()),  # will be overwritten by fused score if decomp exists
        "vqa_pred_score_1to5": float(vqa_pred_score_1to5.item()),
    }
    if keep_details:
        out.update(
            {
                "yes_logit": float(yes_logit.item()),
                "no_logit": float(no_logit.item()),
                "vqa_margin": float(vqa_margin.item()),
                "vqa_p_yes": float(vqa_p_yes.item()),
                "margin": float(margin.item()),
                "p_yes": float(p_yes.item()),
            }
        )
    decomp_gate = getattr(outputs, "decomp_gate", None)
    if keep_details and decomp_gate is not None:
        out["decomp_gate"] = [float(x) for x in decomp_gate[0].detach().cpu().tolist()]

    final_align_score = getattr(outputs, "final_align_score", None)
    if final_align_score is not None:
        decomp_p_yes = torch.sigmoid(final_align_score)
        out["decomp_pred_score_1to5"] = float((1.0 + 4.0 * decomp_p_yes).item())
        # Fused score in logit space.
        fused_margin = vqa_margin + float(fuse_alpha) * final_align_score
        fused_p_yes = torch.sigmoid(fused_margin)
        out["pred_score_1to5"] = float((1.0 + 4.0 * fused_p_yes).item())
        if keep_details:
            out["final_align_score"] = float(final_align_score.item())
            out["decomp_p_yes"] = float(decomp_p_yes.item())
            out["fused_margin"] = float(fused_margin.item())
            out["fused_p_yes"] = float(fused_p_yes.item())
    return out


def main(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
        preprocessor_path=args.preprocessor_path,
    )
    model.eval()
    yes_id, no_id = resolve_yes_no_ids(model, tokenizer)

    os.makedirs(args.save_dir, exist_ok=True)
    conv_mode = args.conv_mode if args.conv_mode is not None else "mplug_owl2"

    for meta_path in args.meta_paths:
        with open(meta_path, "r", encoding="utf-8") as f:
            iqadata = json.load(f)

        save_path = os.path.join(args.save_dir, os.path.basename(meta_path))
        handled_ids = set()
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    line = line.strip()
                    if line:
                        handled_ids.add(json.loads(line)["id"])

        meta_name = os.path.basename(meta_path)
        gate_sum = None
        gate_count = 0
        with open(save_path, "a", encoding="utf-8") as fw:
            for item in tqdm(iqadata, desc=f"Evaluating [{meta_name}]"):
                if item["id"] in handled_ids:
                    continue

                image = load_image(item["image"])
                image = expand2square(
                    image, tuple(int(x * 255) for x in image_processor.image_mean)
                )
                image_tensor = image_processor.preprocess(
                    image, return_tensors="pt"
                )["pixel_values"].half()

                question = build_question(item["prompt"])
                prompt = build_vqa_prompt(conv_mode, question)
                pred = infer_one(
                    model=model,
                    tokenizer=tokenizer,
                    image_tensor=image_tensor,
                    prompt=prompt,
                    yes_id=yes_id,
                    no_id=no_id,
                    device=args.device,
                    use_align_residual=(not args.disable_align_residual),
                    fuse_alpha=args.fuse_alpha,
                    keep_details=args.debug,
                )

                meta_res = {
                    "id": item["id"],
                    "image": item["image"],
                    "prompt": item["prompt"],
                    "mos_align": item.get("mos_align"),
                    "std_align": item.get("std_align"),
                    "question": question,
                    **pred,
                }
                if args.debug:
                    meta_res["full_prompt"] = prompt

                fw.write(json.dumps(meta_res, ensure_ascii=False) + "\n")
                gate = pred.get("decomp_gate")
                if gate is not None:
                    if gate_sum is None:
                        gate_sum = [0.0 for _ in gate]
                    if len(gate_sum) == len(gate):
                        for i, g in enumerate(gate):
                            gate_sum[i] += float(g)
                        gate_count += 1
                torch.cuda.empty_cache()

        if gate_count > 0 and gate_sum is not None:
            gate_mean = [g / gate_count for g in gate_sum]
            gate_entropy = -sum(g * math.log(max(g, 1e-12)) for g in gate_mean)
            perplexity = math.exp(gate_entropy)
            print(
                f"[Gate usage][{meta_name}] samples={gate_count} mean={gate_mean} "
                f"entropy={gate_entropy:.4f} perplexity={perplexity:.3f}"
            )
        else:
            print(
                f"[Gate usage][{meta_name}] no decomp_gate found in newly evaluated samples."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--preprocessor-path", type=str, default=None)
    parser.add_argument("--meta-paths", type=str, required=True, nargs="+")
    parser.add_argument("--save-dir", type=str, default="results_vqa_current")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conv-mode", type=str, default="mplug_owl2")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable-align-residual", action="store_true")
    parser.add_argument("--fuse-alpha", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
