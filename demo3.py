import argparse
import os
import random
import csv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from skingpt4.common.config import Config
from skingpt4.common.dist_utils import get_rank
from skingpt4.common.registry import registry
from skingpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from skingpt4.datasets.builders import *
from skingpt4.models import *
from skingpt4.processors import *
from skingpt4.runners import *
from skingpt4.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

# Process images and save results
def process_images(image_folder, chat, output_csv):
    """Process all images in a folder and save results to a CSV."""
    # Prepare the conversation template
    conv = CONV_VISION.copy()
    img_list = []
    results = []

    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
            image_path = os.path.join(image_folder, image_file)
            print(f"Processing: {image_path}")

            try:
                # Open the image using PIL
                raw_image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
            
            # Upload the image and ask the question
            llm_message = chat.upload_img(raw_image, conv, img_list)
            chat.ask("Describe this condition", conv)

            # Get the model's answer
            response, _ = chat.answer(conv, img_list=[], max_new_tokens=300)

            # Store the result
            results.append({"Image": image_file, "Description": response})

    # Write the results to a CSV file
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Image", "Description"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_csv}")
# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')
print('Processing Images')
IMAGE_FOLDER = "images"  # Update this with your folder path
OUTPUT_CSV = "output_results.csv"
process_images(IMAGE_FOLDER, chat, OUTPUT_CSV)
print('Processed Images')




