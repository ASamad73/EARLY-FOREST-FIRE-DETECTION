Our project began with establishing a baseline using CLIP's zero-shot classification capabilities. We tested three prompt strategies of varying complexity across multiple datasets. Simple prompts achieved only 35% accuracy on augmented data and 23% on original images, demonstrating their ineffectiveness. More detailed prompts showed significant improvement, reaching 80-88% accuracy, confirming that richer contextual prompts yield better results. Interestingly, non-augmented data consistently outperformed augmented versions in our zero-shot inference tests, suggesting that augmentation techniques may introduce noise when used during inference rather than training.

The first major improvement involved developing a two-stage pipeline combining YOLOv8 and CLIP. Initial tests revealed YOLOv8 performed better on general fire images compared to distant forest fire scenes, which presented greater detection challenges due to subtle visual cues. Following professor recommendations, we enhanced the system by integrating YOLOv8 and CLIP for joint inferences. This integrated approach showed notable performance improvements, particularly for the challenging forest fire dataset.

For our second improvement, we implemented several advanced techniques across different implementations. The classifier version used YOLOv8 for localization and CLIP for verification, balanced through a logistic classifier with weights alpha=0.6 and beta=0.4. The padding variation retained more contextual information by applying 10% padding during classifier training. We also experimented with simplified prompts of the same categories but shorter length, and tested different weight configurations where CLIP's influence (beta) was increased relative to YOLO's (alpha). These refinements allowed us to optimize the balance between precise localization and semantic understanding for improved fire detection performance.

It is to note that each folder has its own seperate readme file as well that explains that part.
<hr/>

<h1 align="center">🔥 Early Forest Fire Detection 🔥</h1>
<p align="center">
  <b>YOLOv8 + CLIP based Early Wildfire Detection System</b><br>
  <i>Multi-stage Deep Learning Pipeline to Enhance Detection in Forested Environments</i>
</p>

<p align="center">
  <a href="https://github.com/ASamad73/Early-Forest-Fire-Detection">🔗 GitHub Repo</a> |
  <a href="https://github.com/ASamad73/Early-Forest-Fire-Detection/blob/main/Early-Forest-Fire-Detection(DL%20Project).pdf">📄 Research Paper</a>
</p>

---

## 📌 Project Overview

Early detection of forest fires is vital for minimizing environmental and economic damage. Traditional detection methods lack scalability and sensitivity, especially in challenging conditions. In this project, we explore a deep learning-based multi-stage pipeline using:

- **CLIP (Contrastive Language-Image Pretraining)** for semantic classification
- **YOLOv8** for object detection and localization

The system was iteratively improved in 3 stages:

1. **Baseline**: Zero-shot classification using CLIP
2. **Improvement 1**: Two-stage detection (YOLO → CLIP and CLIP → YOLO)
3. **Improvement 2**: Integrated YOLOv8 + CLIP pipeline with logistic head and weighted fusion

---

## 🧠 Architecture & Methodology

### 🔹 Baseline: Zero-Shot CLIP Classification

- Utilized CLIP model for binary image classification (fire vs non-fire)
- Tried various prompt complexities (simple → moderate → detailed)
- Applied augmentations like:
  - Brightness/contrast changes
  - Pixel value scaling
  - BGR to RGB conversion

> **Best Accuracy:** 88.00% with detailed prompts (unaugmented)

---

### 🔹 Improvement 1: Two-Stage Pipelines

**Method 1: YOLO → CLIP**
- Trained YOLOv8 on annotated fire datasets
- Detected regions passed to CLIP for severity analysis

**Method 2: CLIP → YOLO**
- CLIP filtered input images before passing them to YOLO
- Reduced false positives and computational load

> **Best mAP@0.5:** 89.3% on General Fire Dataset

---

### 🔹 Improvement 2: Integrated Pipeline (YOLO + CLIP)

- Both models fused in a **single weighted pipeline**
- Added logistic head to balance YOLO confidence with CLIP semantic score
- Fine-tuned with padded and non-padded crops
- Fusion formula: `score = α * YOLO_conf + β * CLIP_prob`

> **Best Results (No Padding):**  
> Precision: 0.9134 | Recall: 0.5647 | F1 Score: 0.6979

---

## 📊 Results Summary

| Pipeline                      | Precision | Recall | F1 Score | Accuracy |
|------------------------------|-----------|--------|----------|----------|
| CLIP Zero-Shot (Detailed)    | 88.00%    | -      | -        | -        |
| YOLO+CLIP (2-stage)          | 93.2%     | 86.1%  | -        | -        |
| YOLO+CLIP (Integrated, Padded) | 80.35%    | 69.67% | 74.63%   | 71.06%   |
| YOLO+CLIP (Integrated, No Padding) | 91.34%    | 56.47% | 69.79%   | 67.93%   |
| YOLO+CLIP (α=0.3, β=0.7)     | 36.37%    | 88.58% | 51.57%   | 43.86%   |

---

## 🗂️ Datasets Used

| Dataset | Source | Description |
|--------|--------|-------------|
| Mendeley Fire | [Link](https://data.mendeley.com/datasets/gjmr63rz2r/1) | Unannotated baseline data |
| Roboflow (Wildfire/Forest) | [Link](https://universe.roboflow.com/waleed-azzi-o5bzp/wildfire-detection-3vcvr) | Annotated fire/smoke dataset |
| Alik & Kutay Datasets | [Link](https://www.kaggle.com/datasets/alik05/forest-fire-dataset) | Mixed fire and non-fire scenes |
| General Fire Dataset | [Link](https://universe.roboflow.com/situational-awarnessinnovsense/fire-detection-ypseh) | Used for additional diversity |

---

## 📷 Visual Examples

<p align="center">
  <img src="https://user-images.githubusercontent.com/your-image-link-here" alt="Prediction Example" width="600">
</p>

---

## 🛠️ Tools & Libraries

- Python, PyTorch, OpenCV
- CLIP by OpenAI
- YOLOv8 via Ultralytics
- Roboflow for dataset generation
- Matplotlib, Seaborn for visualization

---

## 📌 Key Takeaways

- Detailed prompts significantly improve CLIP’s zero-shot accuracy
- Sequential pipelines are effective but slow
- Integrated models with weight tuning balance recall and precision
- Padding and prompt engineering enhance classification reliability

---

## 🧪 Future Work

- Add adaptive α/β weighting based on input type
- Incorporate temporal data (video feed analysis)
- Deploy as an edge-based or mobile monitoring tool

---

## 👨‍💻 Authors

- Abdul Samad – [GitHub](https://github.com/ASamad73)
- Haider Abbas Virk

---

## 📄 Reference

The research work and detailed metrics are documented in our [research paper](https://github.com/ASamad73/Early-Forest-Fire-Detection/blob/main/Early-Forest-Fire-Detection(DL%20Project).pdf).

---

⭐️ *If you found this project helpful, please give it a star!*
