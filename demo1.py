import os
import csv
from skingpt4.common.config import Config
from skingpt4.common.registry import registry
from skingpt4.conversation.conversation import Chat, CONV_VISION


def initialize_chat(cfg_path, gpu_id):
    """Initialize the Chat instance."""
    print('Initializing Chat')
    cfg = Config({"cfg_path": cfg_path})

    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(gpu_id))
    print('Initialization Finished')
    return chat


def process_images(image_folder, chat, output_csv):
    """Process all images in a folder and save results to a CSV."""
    # Prepare the conversation template
    conv = CONV_VISION.copy()
    results = []

    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
            image_path = os.path.join(image_folder, image_file)
            print(f"Processing: {image_path}")

            # Upload the image and ask the question
            chat.upload_img(image_path, conv, img_list=[])
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


if __name__ == "__main__":
    # Configuration file path and GPU ID
    CFG_PATH = "eval_configs/skingpt4_eval_vicuna.yaml"  # Update this with your config file path
    GPU_ID = 0  # Update this if using a different GPU

    # Input image folder and output CSV file
    IMAGE_FOLDER = "images"  # Update this with your folder path
    OUTPUT_CSV = "output_results.csv"  # Update this with your desired output CSV file path

    # Initialize the Chat instance
    chat_instance = initialize_chat(CFG_PATH, GPU_ID)

    # Process the images and save results
    process_images(IMAGE_FOLDER, chat_instance, OUTPUT_CSV)
