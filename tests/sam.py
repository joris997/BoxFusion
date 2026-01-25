import torch
from PIL import Image
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from PIL import Image
import clip
from typing import Any, Dict, List, Optional

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- SAM ---
sam = sam_model_registry["vit_h"](
    checkpoint="BoxFusion/models/sam_vit_h_4b8939.pth"
).to(device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.96,
    min_mask_region_area=800,
)

# --- CLIP ---
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()


def detect_objects(
    image_path: str,
    object_names: List[str],
) -> List[Optional[Dict[str, Any]]]:
    # Load image
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Generate masks once per image
    masks = mask_generator.generate(image_rgb)

    # Encode all text prompts once
    texts = [f"a photo of a {name}" for name in object_names]
    text_tokens = clip.tokenize(texts).to(device)

    with torch.no_grad():
        text_feats = clip_model.encode_text(text_tokens)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)

    best_scores: List[float] = [-1.0 for _ in object_names]
    best_masks: List[Optional[Dict[str, Any]]] = [None for _ in object_names]

    # Score every mask against every prompt
    for m in masks:
        x, y, w, h = map(int, m["bbox"])

        # Reject degenerate boxes early
        if w < 20 or h < 20:
            continue

        crop = image_rgb[y:y + h, x:x + w]
        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(crop)
        crop_tensor = clip_preprocess(crop_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            img_feat = clip_model.encode_image(crop_tensor)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)

            scores = (img_feat @ text_feats.T).squeeze(0)

        # Update best match per prompt
        for idx, score in enumerate(scores.tolist()):
            if score > best_scores[idx]:
                best_scores[idx] = score
                best_masks[idx] = m

    results: List[Optional[Dict[str, Any]]] = []
    for name, best_score, best_mask in zip(object_names, best_scores, best_masks):
        if best_mask is None:
            results.append(None)
            continue

        bx, by, bw, bh = map(int, best_mask["bbox"])
        results.append(
            {
                "object_name": name,
                "bbox": (bx, by, bw, bh),
                "score": best_score,
            }
        )

    return results

image_path = os.path.expanduser("~/Pictures/Screenshots/test_image.png")
image = Image.open(image_path).convert("RGB")
object_names = [
    "measurement cup filled with broccoli",
    "bowl with rice",
    "bell pepper",
    "tomato",
    "broccoli",
    "bottle of mustard"
]

results = detect_objects(
    image_path=image_path,
    object_names=object_names,
)

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.imshow(image)

for name, result in zip(object_names, results):
    if result is None:
        print(f"No detection for {name}")
        continue

    x, y, w, h = result["bbox"]
    score = result["score"]

    print(f"Detected {name} @ {result['bbox']}  score={score:.3f}")
    rect = plt.Rectangle(
        (x, y),
        w, h,
        fill=False,
        color="green",
        linewidth=2,
    )
    ax.add_patch(rect)
    ax.text(
        x, y-6,
        f"{name} ({score:.2f})",
        fontsize=16,
        color="green",
    )
plt.axis("off")
plt.show()

    # img = cv2.imread("image.png")
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # cv2.putText(
    #     img,
    #     f"mug ({score:.2f})",
    #     (x, y-6),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.6,
    #     (0,255,0),
    #     2
    # )
    # cv2.imwrite("result.png", img)


# # # Load a model
# # model = YOLO("yolov8n.pt", task="detect")

# # # detect tomato
# # results = model.predict(source=image, conf=0.25, save=False)

# # # show results
# # results[0].show()

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Grounding DINO (text → boxes)
# dino_processor = AutoProcessor.from_pretrained(
#     "IDEA-Research/grounding-dino-base"
# )
# dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
#     "IDEA-Research/grounding-dino-base"
# ).to(device)

# # SAM (boxes → masks)
# sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
# sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

# text_prompt = "bell pepper"


# inputs = dino_processor(
#     images=image,
#     text=text_prompt,
#     return_tensors="pt"
# ).to(device)

# with torch.no_grad():
#     outputs = dino_model(**inputs)

# results = dino_processor.post_process_grounded_object_detection(
#     outputs,
#     inputs.input_ids,
#     threshold=0.2,
#     text_threshold=0.1,
#     target_sizes=[image.size[::-1]],  # (H, W)
# )

# boxes = results[0]["boxes"]  # (N, 4) in pixel coords
# print(f"Detected {boxes.shape[0]} boxes for prompt '{text_prompt}'")

# # display image and the box
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 1, figsize=(16, 10))
# ax.imshow(image)
# for box in boxes:
#     box = box.cpu().numpy()
#     x0, y0, x1, y1 = box
#     rect = plt.Rectangle(
#         (x0, y0),
#         x1 - x0,
#         y1 - y0,
#         fill=False,
#         color="green",
#         linewidth=2,
#     )
#     ax.add_patch(rect)
# plt.axis("off")
# plt.show()