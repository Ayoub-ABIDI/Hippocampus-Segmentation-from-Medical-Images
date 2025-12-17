# Hippocampus Segmentation from Medical Images

A comprehensive computer vision project exploring multiple deep learning architectures for automated hippocampus segmentation from 3D MRI images.

**Author:** Ayoub Abidi  
**Institution:** National Engineering School of Tunis  
**Program:** 3rd Year DSIC Student

<img width="878" height="597" alt="image" src="https://github.com/user-attachments/assets/1f0e5a63-ea43-427c-8044-ad47c870fd40" />


---

## 📋 Table of Contents

- [Motivation & Objective](#motivation--objective)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architectures](#model-architectures)
- [Results Comparison](#results-comparison)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Key Findings](#key-findings)

---

## 🎯 Motivation & Objective

### Objective
Automated segmentation of the hippocampus from medical images using deep learning approaches.

### Goal
Experiment with different architectures to identify the most effective approach for hippocampus segmentation.

### Focus
- Comparison and analysis of multiple segmentation methods
- Evaluation of 2D vs 3D approaches
- Integration of attention mechanisms
- Exploration of vision-language models for medical imaging

### Importance
- Advances precision in medical imaging
- Improves research reproducibility
- Provides benchmark comparisons for different architectures

---

## 📊 Dataset

**Dataset Overview:**
- 3D MRI images for hippocampus segmentation
- **Classes:** 3 (Background, Anterior Hippocampus, Posterior Hippocampus)
- **Training Set:** 260 volumes
- **Test Set:** 130 volumes
- **Total 2D Slices (after conversion):** 16,640

**Dataset Challenges:**
- **Variable Image Sizes:** Images have different dimensions requiring preprocessing
- **Intensity Variation:** Intensity values vary across volumes
- **Class Imbalance:** Background dominates (~95%), Hippocampus regions are small (~5%)

---

## 📁 Project Structure

```
Task04_Hippocampus/
├── imagesTr/              # Training images
├── imagesTs/              # Test images
├── labelsTr/              # Training labels
├── preprocessed/          # Preprocessed data
├── results/               # Model outputs and predictions
├── dataset.json           # Dataset configuration
├── preprocessing_utils.py # Preprocessing utilities
│
├── 1_exploration.ipynb           # Data exploration and analysis
├── 2_preprocessing.ipynb         # Preprocessing pipeline
├── 3_unet_2d_baseline.ipynb     # 2D U-Net experiments
├── 4_UNet3D.ipynb               # 3D U-Net implementation
├── 5_AttentionUNet.ipynb        # Attention U-Net experiments
├── 6_VLM_Guided_Segmentation.ipynb  # CLIP-guided U-Net
│
├── class_distribution.png
├── dimensions_distribution.png
├── example_visualization.png
├── multi_axis_view.png
└── preprocessing_comparison.png
```

---

## 🔧 Preprocessing Pipeline

### 1. Intensity Normalization
- Applied **z-score normalization** to all volumes
- Ensures intensities are centered and scaled consistently
- Helps models learn features more effectively

### 2. Padding
- Applied padding to reach uniform size: **64 × 64 × 64**
- Ensures all images and labels match model input requirements
- Maintains spatial relationships

### 3. Handling Class Imbalance
- Used **Dice Loss** and **Focal Loss** during training
- Addresses the severe class imbalance (95% background vs 5% hippocampus)

---

## 🧠 Model Architectures

### 1. **2D U-Net (Baseline)**
- Converted 3D volumes into 2D slices for training
- Total slices: 16,640
- Explored multiple data augmentation strategies

**Augmentation Experiments:**
| Experiment | Augmentation Strategy | Dice Score | Training Time |
|------------|----------------------|------------|---------------|
| 1 | No augmentation | **0.9225** | - |
| 2 | 50% flip | 0.9176 | - |
| 3 | Flip + 30% rotation | 0.9178 | - |
| 4 | Flip + rotation + noise | 0.9138 | - |

**Key Insight:** Surprisingly, no augmentation performed best, possibly due to sufficient dataset size.

---

### 2. **3D U-Net**
- Processes entire volume at once (not slice-by-slice)
- Better captures 3D spatial context
- More appropriate for 3D anatomical structures like the hippocampus

**Performance:**
- **Dice Score:** 0.8108
- **Training Time:** 32 minutes

**Challenge:** Lower performance than 2D U-Net, possibly due to:
- Increased model complexity
- Limited training data for 3D approach
- Need for more extensive hyperparameter tuning

---

### 3. **Attention U-Net**
- Base architecture: U-Net with **attention gates**
- Attention gates filter out irrelevant features
- Focuses specifically on hippocampus regions

**How Attention Gates Work:**
- Receives encoder features (spatial details) and decoder features (what we're looking for)
- Computes attention coefficients between 0 and 1
- Important regions → high weight
- Irrelevant regions → suppressed
- Encoder features are filtered before fusion with decoder

**Performance:**
- **Dice Score:** 0.9090
- **Training Time:** 67 minutes

**Benefit:** Improved performance on small, complex structures by focusing on relevant regions.

---

### 4. **VLM-Guided U-Net (CLIP Integration)**

#### What are Vision-Language Models (VLMs)?
Models trained to jointly understand images and text, learning semantic meaning associated with visual content.

#### CLIP (Contrastive Language-Image Pretraining)
- Developed by OpenAI
- Trained on large-scale image-text pairs
- Learns shared embedding space for text and images
- Uses contrastive learning

#### Why Use CLIP for Segmentation?
- Classical U-Net learns only from pixel-level information
- Hippocampus is small and easily confused with background tissue
- CLIP injects **semantic knowledge** into the network
- Guides the model toward regions corresponding to "hippocampus" concept

#### Architecture Overview

**Step 1:** Image input fed to both U-Net encoder and CLIP visual encoder

**Step 2:** CLIP visual encoder (frozen) extracts high-level semantic features

**Step 3:** Feature projection aligns CLIP features with U-Net feature space

**Step 4:** Fusion at U-Net bottleneck combines:
- Local spatial information (from U-Net encoder)
- Global semantic information (from CLIP)

**Step 5:** Decoder upsamples fused features with skip connections

**Step 6:** Pixel embedding replaces standard classifier
- Each pixel projected into CLIP embedding space
- Represented as semantic vector

**Step 7:** Cosine similarity computation
- Between pixel embeddings and class text embeddings
- Higher similarity = higher probability for that class

**Step 8:** Segmentation logits and final prediction

**Key Innovation:** U-Net handles spatial precision, CLIP provides semantic understanding, and similarity with text embeddings replaces traditional classifier.

**Performance:**
- **Dice Score:** 0.9084
- **Training Time:** 235 minutes

---

## 📈 Results Comparison

| Model | Dice Score | Training Time | Key Characteristics |
|-------|------------|---------------|---------------------|
| **2D U-Net (No Aug)** | **0.9225** | - | Best overall performance, treats slices independently |
| 2D Attention U-Net | 0.9090 | 67 min | Focuses on relevant regions, good for small structures |
| CLIP-Guided U-Net | 0.9084 | 235 min | Semantic guidance, most interpretable |
| 2D U-Net (50% flip) | 0.9176 | - | Good performance with augmentation |
| 3D U-Net | 0.8108 | 32 min | Captures 3D context but needs more tuning |

---

## 🔍 Key Findings

1. **2D U-Net performed best** despite not using 3D spatial information
2. **Attention mechanisms** significantly help with small structure segmentation
3. **VLM-guided approaches** show promise for semantic understanding in medical imaging
4. **3D U-Net underperformed**, suggesting need for:
   - More training data
   - Better hyperparameter tuning
   - Possibly deeper architectures
5. **Data augmentation** didn't improve results, possibly due to sufficient dataset size

---

## 💻 Installation & Setup

### Prerequisites
```bash
Python 3.8+
PyTorch 1.10+
CUDA (for GPU support)
```

### Install Dependencies
```bash
# Clone the repository
git clone https://github.com/yourusername/hippocampus-segmentation.git
cd hippocampus-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install torch torchvision
pip install numpy pandas matplotlib
pip install nibabel scikit-image
pip install jupyter notebook
pip install clip-by-openai  # For VLM experiments
```

---

## 🚀 Usage

### 1. Data Exploration
```bash
jupyter notebook 1_exploration.ipynb
```
Explore dataset characteristics, visualize samples, and analyze class distribution.

### 2. Preprocessing
```bash
jupyter notebook 2_preprocessing.ipynb
```
Run preprocessing pipeline: normalization, padding, and data preparation.

### 3. Train Models

**2D U-Net:**
```bash
jupyter notebook 3_unet_2d_baseline.ipynb
```

**3D U-Net:**
```bash
jupyter notebook 4_UNet3D.ipynb
```

**Attention U-Net:**
```bash
jupyter notebook 5_AttentionUNet.ipynb
```

**CLIP-Guided U-Net:**
```bash
jupyter notebook 6_VLM_Guided_Segmentation.ipynb
```

---

## 📚 References

- **U-Net:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- **Attention U-Net:** Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas" (2018)
- **CLIP:** Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021)

---

## 📝 License

This project is part of academic research at the National Engineering School of Tunis.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

## 👤 Contact

**Ayoub Abidi**  
National Engineering School of Tunis  
3rd Year DSIC Student

---

## 🙏 Acknowledgments

- National Engineering School of Tunis
- Medical Segmentation Decathlon Dataset
- OpenAI CLIP Team
