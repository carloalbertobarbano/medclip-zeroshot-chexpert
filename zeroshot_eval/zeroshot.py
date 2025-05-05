import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from medclip import MedCLIPModel, MedCLIPVisionModelViT
from transformers import AutoTokenizer

from typing import List, Tuple

import torch
from torch.utils import data
from tqdm import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

from .eval import evaluate


def evaluate_cxr_zeroshot(
    clip_model: MedCLIPModel,
    val_cxr_loader,
    cxr_labels: list[str] = None,
    cxr_sub_labels: list[str] = None,
    plot=False,
):
    assert cxr_labels is not None
    cxr_pair_template: tuple[str] = ("{}", "no {}")
    y_pred, y_true = run_softmax_eval(
        clip_model, val_cxr_loader, cxr_labels, cxr_pair_template
    )
    cxr_results = evaluate(y_pred, y_true, cxr_labels, cxr_sub_labels, plot=plot)
    return cxr_results


def zeroshot_classifier(model: MedCLIPModel, classnames, templates, device="mps"):
    """
    FUNCTION: zeroshot_classifier
    -------------------------------------
    This function outputs the weights for each of the classes based on the
    output of the trained clip model text transformer.

    args:
    * model - Pytorch model, full trained clip model.
    * classnames - Python list of classes for a specific zero-shot task. (i.e. ['Atelectasis',...]).
    * templates - Python list of phrases that will be indpendently tested as input to the clip model.

    Returns PyTorch Tensor, output of the text encoder given templates.
    """
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    tokenizer.model_max_length = 77

    with torch.no_grad():
        zeroshot_weights = []
        # compute embedding through model for each class
        for classname in tqdm(classnames):
            texts = [
                template.format(classname) for template in templates
            ]  # format with class

            texts = tokenizer(texts, return_tensors="pt")['input_ids'].to(device) # clip.tokenize(texts, truncate=True).cuda()  # tokenize
            # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            class_embeddings = model.encode_text(texts)  # embed with text encoder

            # normalize class_embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # average over templates
            class_embedding = class_embeddings.mean(dim=0)
            # norm over new averaged templates
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def run_prediction(model: MedCLIPModel, pos_prompts_encodings, neg_prompts_encodings, loader, device="mps"):
    """

    Returns nd arrays.
    """
    pos_pred, neg_pred, y_true = [], [], []
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader)):
            images = images.to(device)
            print(labels.shape)
            # print(images.shape)
            # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = model.encode_image(images)

            image_features /= image_features.norm(dim=-1, keepdim=True)  # (1, 768)

            # obtain logits
            logit_scale = model.logit_scale.exp()
            # Think about using the logits
            pos_logits = image_features @ pos_prompts_encodings * logit_scale
            neg_logits = image_features @ neg_prompts_encodings * logit_scale

            # logits = np.squeeze(logits.numpy(), axis=0) # (num_classes,)

            pos_pred.append(pos_logits)
            neg_pred.append(neg_logits)
            y_true += labels

            """if verbose: 
                plt.imshow(images[0][0])
                plt.show()
                print('images: ', images)
                print('images size: ', images.size())
                
                print('image_features size: ', image_features.size())
                print('logits: ', logits)
                print('logits size: ', logits.size())"""

    pos_pred = torch.cat(pos_pred).detach().cpu().numpy()
    neg_pred = torch.cat(neg_pred).detach().cpu().numpy()
    print("y_true:", y_true)
    # print("y_true:", y_true.shape)
    y_true = torch.stack(y_true).detach().cpu().numpy()
    # y_true = np.array(y_true, dtype=int)
    print("pos_pred", pos_pred.shape, "neg_pred", neg_pred.shape, "y_true:", y_true.shape)
    return pos_pred, neg_pred, y_true


def run_softmax_eval(model: MedCLIPModel, loader, eval_labels: list, pair_template: tuple):
    """
    Run softmax evaluation to obtain a single prediction from the model.
    """
    # get pos and neg phrases
    pos = pair_template[0]
    neg = pair_template[1]

    # compute the zeroshot prompts encodings
    pos_prompts_encodings = zeroshot_classifier(model, eval_labels, [pos])
    neg_prompts_encodings = zeroshot_classifier(model, eval_labels, [neg])

    # get pos and neg predictions, (num_samples, num_classes)
    pos_pred, neg_pred, y_true = run_prediction(
        model, pos_prompts_encodings, neg_prompts_encodings, loader
    )
    # compute probabilities with softmax
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred
    return y_pred, y_true
