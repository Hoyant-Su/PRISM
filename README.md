# PRISM: Prompt-guided Representation Integration for Survival Modeling

## Table of Contents

1. [Introduction](#introduction)  
2. [Highlights](#highlights)  
3. [Installation](#installation)  
4. [Datasets and Usage](#datasets-and-usage)  
5. [Model Architecture](#model-architecture)  
6. [Training](#training)  
7. [Evaluation](#evaluation)  

---

## Introduction

- PRISM is a self-supervised framework for predicting major adverse cardiac events (MACE) by integrating cardiac MRI and clinical data. We use motion-aware feature extraction combined with prompt-guided modeling to improve survival analysis accuracy.

## Highlights

- 📍Self-supervised framework  
- 📍Multi-view motion-aware feature distillation  
- 📍Prompt-guided modulation using medical knowledge  
- 📍Supports survival analysis with fine-grained risk prediction  
- 📍Validated on multiple clinical cohorts  

## Installation

```bash
git clone https://github.com/Hoyant-Su/PRISM.git
conda create --name PRISM python=3.10
conda activate PRISM
cd PRISM
pip install -r requirements.txt
```

## Datasets and Usage

###  Image Pre-processing: 🫀 Myocardial Localization
We utilize the optical flow algorithm from OpenCV (`cv2`) to extract the position of the myocardium. The script `preprocess/optical_flow_crop.py` provides a function to perform this localization within 4D Nifti data.

### Metadata Preparation: 📝 JSON File Structure
Before running the training script, a JSON file that catalogs the dataset must be prepared. The JSON file should be structured as a list, where each element is a dictionary representing a unique patient case.

The key of each dictionary entry should be the patient or case ID. Paths to all associated image files under the `"img"` key and other required metadata can be organized below.

**Example of `survival_analysis.json`:**

```json
[
  {
    "1703061492": {
      "img": [
        "/path/to/your/data/sample_0/CINE-SA/0.nii.gz",
        "/path/to/your/data/sample_0/CINE-2CH/0.nii.gz"
        "/path/to/your/data/sample_0CINE-3CH/0.nii.gz"
        "/path/to/your/data/sample_0CINE-4CH/0.nii.gz"
      ],
      "performance": "wo",
      "diagnosis": "wo",
      "survival_time": 93.73333333333336,
      "c": "1",
      "center": "cohortA"
    }
  },
  {
    "1707100347": {
      "img": [
        "/path/to/your/data/sample_1/CINE-SA/0.nii.gz",
        "/path/to/your/data/sample_1/CINE-2CH/0.nii.gz"
        "/path/to/your/data/sample_1/CINE-3CH/0.nii.gz"
        "/path/to/your/data/sample_1/CINE-4CH/0.nii.gz"
      ],
      "performance": "wo",
      "diagnosis": "wo",
      "survival_time": 89.66666666666669,
      "c": "1",
      "center": "cohortB"
    }
  }
]
```

## Model Architecture

The PRISM model is trained in three distinct stages:

-   **Stage I: ⚙️ Motion-Aware Multi-View Distillation**
    -   Knowledge distillation using multi-view CMR data (SAX & LAX at 2/3/4CH).

-   **Stage II: ⚙️ EHR-Attention Guidance**
    -   An attention mechanism guided by Electronic Health Records (EHR).

-   **Stage III: ⚙️ Survival Prediction**
    -   Fine-tuning for survival analysis using a Cox Proportional-Hazards (CoxPH) model.

![PRISM Framework](assets/PRISM_framework.png)


## Training

The model is trained in a three-stage process. This includes unsupervised learning for stages I & II (code available in `PRISM/ssl_pretraining`) and survival analysis fine-tuning for stage III (code available in `PRISM/finetune_survival`).

Basic configurations should be prepared with .yaml files before training in `configs` folder.


To start the training process, run the following script:

```bash
bash launch_train.sh
```

## Evaluation

After the training process is complete, the following outputs are generated:

-   **Model Checkpoints**: The checkpoints from the SSL pre-training are automatically stored in `ssl_pretraining/weights`.
-   **Survival Analysis Results**: All evaluation results and logs for the survival analysis are saved to `finetune_survival/logs`.`

