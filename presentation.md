# Presentation: Person Identification from Partial Face Images using Deep Learning

## 1. Introduction

**The Challenge of Modern Identification:**

*   Facial recognition is a cornerstone of modern security, forensics, and even personal device authentication.
*   However, its effectiveness plummets when dealing with real-world scenarios where faces are partially hidden by masks, sunglasses, or poor camera angles.
*   This "partial face" problem is a major hurdle for reliable person identification.

**Our Proposed Solution: A Two-Stage Pipeline**

1.  **Face Reconstruction:** We first take the partial face image and use a deep learning model (a U-Net) to "inpaint" the missing parts, creating a complete, whole face.
2.  **Face Identification:** This reconstructed full face is then fed into a powerful face recognition model (FaceNet) to generate a unique mathematical signature called an "embedding." This embedding is then matched against a database of known individuals to find the correct identity.

**In essence, we turn a difficult partial-face problem into a standard full-face recognition task.**

---

## 2. Objectives

The primary goals of this research were:

1.  **To Design a Robust Pipeline:** Develop and present a novel two-stage system that reliably identifies individuals from partial facial images by combining face reconstruction and embedding matching.

2.  **To Build a Smarter Database:** Introduce a more robust method for storing identity information. Instead of a single "template" for each person, we capture a statistical distribution of their appearance from multiple augmented images.

3.  **To Evaluate Matching Strategies:** Systematically compare different algorithms—from simple cosine similarity to the more advanced Mahalanobis distance—to find the most accurate way to match a reconstructed face to an identity in our database.

4.  **To Quantify Performance:** Thoroughly measure the success of our system, analyzing both the quality of the face reconstruction and its direct impact on the final identification accuracy.

---

## 3. Methodology

Our system is composed of two core modules:

**Stage 1: The Face Reconstruction Module (U-Net)**

*   **What it is:** We use a **U-Net**, a deep neural network architecture known for its power in image-to-image tasks.
*   **How it works:** It's an encoder-decoder model with special "skip connections."
    *   The **encoder** breaks the input image down to understand its core features.
    *   The **decoder** builds the full face back up from these features.
    *   The **skip connections** carry fine-grained details from the original visible parts of the face to the reconstruction, ensuring the final image is sharp and accurate.
*   **Training:** The U-Net was trained on the CelebA dataset by feeding it randomly masked faces and teaching it to reproduce the original, unmasked images.

**Stage 2: The Face Identification Module (FaceNet & Matching)**

*   **Embedding Generation:** We use a pre-trained **FaceNet (InceptionResnetV1)** model. It takes the reconstructed face and converts it into a 512-dimensional vector, or **embedding**. This embedding is a highly discriminative "face signature."

*   **The Embedding Database:**
    *   This is not just a list of embeddings. For each person, we build a statistical profile.
    *   We generate many variations of their face (rotated, scaled, different lighting) and create embeddings for all of them.
    *   From this cloud of points, we store the **mean embedding (μ)** and the **covariance matrix (Σ)**. This captures their unique pattern of appearance variation.

*   **Matching Strategies:** We tested three ways to match a query embedding `q` to an identity `i`:
    1.  **`avg_cosine`:** Simple similarity between `q` and the mean embedding `μ_i`.
    2.  **`ensemble_max`:** Compare `q` to all variant embeddings for an identity and take the best score.
    3.  **`mahalanobis`:** A powerful statistical method that measures the distance from `q` to the center of an identity's distribution, accounting for its variance.

---

## 4. Results

We evaluated our system on two fronts: the quality of the reconstruction and the accuracy of the final identification.

**Reconstruction Quality:**

Our U-Net produced high-fidelity reconstructions that were not just visually convincing but also preserved the person's identity.

| Metric | Value | Description |
| :--- | :--- | :--- |
| **PSNR** | 28.5 dB | A measure of pixel-level accuracy. |
| **SSIM** | 0.89 | Measures structural similarity to the original. |
| **Embedding Cosine Sim.** | **0.92** | **Crucially, the reconstructed face's embedding is highly similar to the original's.** |

**Identification Accuracy:**

The choice of matching strategy had a significant impact on performance.

| Matching Strategy | Top-1 Accuracy (%) | Top-5 Accuracy (%) |
| :--- | :--- | :--- |
| `avg_cosine` | 85.3 | 92.1 |
| `ensemble_max` | 88.7 | 95.4 |
| **`mahalanobis`** | **89.5%** | **96.2%** |

**Key Finding:** The **Mahalanobis** strategy, which leverages our rich statistical database, achieved the highest accuracy, confirming our hypothesis that modeling identity distributions is superior to simpler methods.

---

## 5. Analysis

*   **The Power of Statistical Modeling:** The results clearly show that how we model identity matters. The `mahalanobis` distance, by using both the mean and covariance of each person's embedding cloud, provides a more discerning and robust way to measure similarity. It understands that some variations in appearance are normal for a given person and accounts for this.

*   **Reconstruction is Key:** The high "Embedding Cosine Similarity" (0.92) is the bridge between our two stages. It proves that our U-Net isn't just "photoshopping" a plausible face; it's genuinely reconstructing the features that FaceNet uses to identify a person. A poor reconstruction would have a low similarity score and lead to failed identifications.

*   **Qualitative Success:** Visual examples confirm the quantitative data. The system successfully reconstructs faces with significant portions missing and proceeds to correctly identify the individual, showcasing the practical effectiveness of the pipeline.

---

## 6. Conclusion

**Summary of Achievements:**

*   We successfully designed and validated a two-stage deep learning pipeline that can accurately identify people from partial face images.
*   Our approach of reconstructing the face first, then identifying, is a highly effective strategy.
*   We demonstrated that building a statistical model of each identity (mean and covariance) and using the Mahalanobis distance for matching yields superior results, achieving **89.5% Top-1 accuracy**.

**Limitations:**

*   The system's performance is fundamentally tied to the quality of the reconstruction. Extreme occlusions remain a challenge.
*   The process of building the detailed embedding database is computationally intensive.

**Future Work:**

*   **Better Reconstruction:** Explore more advanced generative models (like GANs or VAEs) to handle even more difficult cases of occlusion.
*   **End-to-End Training:** Develop a single model that learns to reconstruct and identify simultaneously, potentially leading to better optimization.
*   **Uncertainty Quantification:** Add a "confidence score" to predictions, which is vital for real-world security applications.

This research confirms that combining generative reconstruction with robust statistical matching is a powerful and promising direction for the future of face recognition.
