from PIL import Image

import torch.nn as nn
import torch

from typing import List

from src.model.builder import load_pretrained_model

from src.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from src.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


class Scorer(nn.Module):
    def __init__(self, pretrained="zhiyuanyou/DeQA-Score-Mix3", device="cuda:0"):
        super().__init__()
        tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, "mplug_owl2", device=device)
        prompt = "USER: How would you rate the quality of this image?\n<|image|>\nASSISTANT: The quality of the image is"

        self.preferential_ids_ = [id_[1] for id_ in tokenizer(["excellent","good","fair","poor","bad"])["input_ids"]]
        self.weight_tensor = torch.Tensor([5.,4.,3.,2.,1.]).half().to(model.device)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    def expand2square(self, pil_img, background_color):
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

    def forward(self, image: List[Image.Image]):
        image = [self.expand2square(img, tuple(int(x*255) for x in self.image_processor.image_mean)) for img in image]
        with torch.inference_mode():
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(self.model.device)
            output_logits = self.model(
                        input_ids=self.input_ids.repeat(image_tensor.shape[0], 1),
                        images=image_tensor
                    )["logits"][:,-1, self.preferential_ids_]

            return torch.softmax(output_logits, -1) @ self.weight_tensor

class ConditionalScorer(nn.Module):
    def __init__(self, pretrained="zhiyuanyou/DeQA-Score-Mix3", device="cuda:0"):
        super().__init__()
        tokenizer, model, image_processor, _ = load_pretrained_model(
            pretrained, None, "mplug_owl2", device=device
        )

        self.preferential_words = ["excellent", "good", "fair", "poor", "bad"]
        self.preferential_ids_ = [
            ids[1] for ids in tokenizer(self.preferential_words)["input_ids"]
        ]
        self.weight_tensor = torch.tensor(
            [5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float16, device=model.device
        )

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

    def build_prompt(self, text_prompt: str) -> str:
        return (
            "USER: Given the text prompt and the image, rate how well the image matches the text prompt.\n"
            f'Text prompt: "{text_prompt}"\n'
            "<|image|>\n"
            "ASSISTANT: The text-image alignment of this image is"
        )

    def expand2square(self, pil_img, background_color):
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

    def forward(self, images: List[Image.Image], text_prompts: List[str]):
        assert len(images) == len(text_prompts), "images and text_prompts must have the same length"

        images = [
            self.expand2square(
                img,
                tuple(int(x * 255) for x in self.image_processor.image_mean)
            )
            for img in images
        ]

        prompts = [self.build_prompt(tp) for tp in text_prompts]
        input_ids = [
            tokenizer_image_token(p, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            for p in prompts
        ]
        input_ids = torch.stack(input_ids, dim=0).to(self.model.device)

        with torch.inference_mode():
            image_tensor = self.image_processor.preprocess(
                images, return_tensors="pt"
            )["pixel_values"].half().to(self.model.device)

            output_logits = self.model(
                input_ids=input_ids,
                images=image_tensor
            )["logits"][:, -1, self.preferential_ids_]

            probs = torch.softmax(output_logits, dim=-1)
            scores = probs @ self.weight_tensor
            return scores, probs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="zhiyuanyou/DeQA-Score-Mix3")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img_path", type=str, default="fig/singapore_flyer.jpg")
    parser.add_argument("--text_prompt", type=str, required=True)
    args = parser.parse_args()

    # scorer = Scorer(pretrained=args.model_path, device=args.device)
    # print(scorer([Image.open(args.img_path)]).tolist())
    scorer = ConditionalScorer(pretrained=args.model_path, device=args.device)
    image = Image.open(args.img_path).convert("RGB")
    scores, probs = scorer([image], [args.text_prompt])

    print("score:", scores.tolist())
    print("probabilities:", probs.tolist())
