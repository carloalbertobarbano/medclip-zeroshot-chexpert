import torch
import torch.utils.data
import os
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore", module="urllib3")

from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from medclip import MedCLIPProcessor

from data.chexpert import ChexpertTest
from zeroshot_eval.zeroshot import evaluate_cxr_zeroshot


def test(model, val_cxr_loader, cxr_labels, cxr_sub_labels, is_sub, device="cuda"):
    results_df = evaluate_cxr_zeroshot(
        clip_model=model,
        val_cxr_loader=val_cxr_loader,
        cxr_labels=cxr_labels,
        cxr_sub_labels=cxr_sub_labels,
        plot=False,
    )

    return results_df


def main():
    set_seed(0)
    device = "mps"
    processor = MedCLIPProcessor()
    # model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained(device=device)
    model.to(device)
    model.eval()

    tot_parameters = sum([np.prod(p.size()) for p in model.parameters()])
    print("Tot. parameters:", tot_parameters)

    cxr_labels: list[str] = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]
    cxr_sub_labels: list[str] = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]

    dataset = ChexpertTest(
        root=os.path.expanduser("~/data/chexpert-test"),
        processor=processor,
        labels=cxr_labels,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=8)

    df = test(model, dataloader, cxr_labels, cxr_sub_labels, is_sub=True, device=device)
    df.to_csv("chexpert_test_results.csv", index=False)
    print(df)

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()