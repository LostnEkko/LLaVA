import argparse
import torch

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from llava.train.train import format_prompt

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, args.concat_projection, args.attn_adapter)

    image = load_image(args.image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    while True:
        try:
            inp = input(f"USER INPUT: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"OUTPUT: ", end="")

        input_ids = torch.tensor(tokenizer.encode(format_prompt(inp, None)), dtype=torch.int64).unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=256,
                use_cache=True
            )
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--concat-projection", action="store_true")
    parser.add_argument("--attn-adapter", action="store_true")
    args = parser.parse_args()
    main(args)
