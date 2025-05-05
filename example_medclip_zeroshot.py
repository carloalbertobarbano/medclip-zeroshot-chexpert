"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 05/05/25
"""
import os
import torch
import random
import numpy as np
from PIL import Image

from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from medclip import PromptClassifier
from medclip.prompts import generate_chexpert_class_prompts, process_class_prompts

device = "mps"

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


@torch.inference_mode()
def run(clf, processor, cls_prompts, image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs['prompt_inputs'] = cls_prompts

    output = clf(**inputs, device=device)
    return output


def main():
    set_seed(0)
    processor = MedCLIPProcessor()
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained(device=device)
    model.eval()
    clf = PromptClassifier(model, ensemble=False)
    clf.to(device)
    clf.eval()

    cls_prompts = generate_chexpert_class_prompts(n=1)
    print(cls_prompts)
    cls_prompts = process_class_prompts(cls_prompts)
    # print(cls_prompts)

    output = run(clf, processor, cls_prompts,
                 image_path=os.path.expanduser("~/data/chexpert-test/test/patient64741/study1/view1_frontal.jpg"))
    print(output)



if __name__ == "__main__":
    main()