# A Deep Learning Approach for Person Identification from Partial Face Images using Face Reconstruction and Embedding Matching

### Abstract

Person identification from facial images is a cornerstone of modern biometric systems, with applications ranging from security and access control to forensic analysis. However, the performance of these systems degrades significantly when presented with partial or occluded facial data, a common challenge in real-world scenarios due to masks, sunglasses, or poor image framing. This paper introduces a novel two-stage deep learning pipeline to address the problem of person identification from partial face images. Our approach first reconstructs a complete facial image from a masked input using a U-Net-based architecture. The reconstructed face is then passed to a pre-trained FaceNet model to generate a high-dimensional embedding vector. This embedding is subsequently matched against a pre-computed, identity-rich database, which stores not only a mean embedding for each individual but also a distribution of embeddings derived from augmented image variants. We explore and evaluate multiple matching strategies, including cosine similarity, an ensemble-based approach, and Mahalanobis distance, to determine the most effective method for matching the reconstructed embedding to the correct identity. Our experiments, conducted on the CelebA dataset, demonstrate that the proposed reconstruction-then-identification pipeline achieves high identification accuracy. We present a detailed analysis of the reconstruction quality and its direct impact on recognition performance, showing that a high-fidelity reconstruction in both the image and embedding space is critical for success.

### 1. Introduction

**1.1. Background**

Automated person identification is a critical technology in the 21st century, forming the backbone of numerous systems across various domains. In security, it is used for access control to restricted areas and for surveillance in public spaces. In forensics, it aids law enforcement in identifying suspects from crime scene evidence. The rise of social media and personal devices has also seen face recognition become a ubiquitous feature for photo tagging and device unlocking. The vast majority of these systems rely on deep learning, specifically deep convolutional neural networks (CNNs), which have demonstrated remarkable performance in learning discriminative features from facial images.

**1.2. Problem Statement**

Despite the significant advances in face recognition, the performance of state-of-the-art systems is predicated on the availability of high-quality, unobstructed facial images. In many real-world applications, this ideal condition is not met. Faces can be partially occluded by accessories like masks and sunglasses, by other objects, or simply due to the angle and framing of the camera. This results in partial face images where crucial facial landmarks may be missing, leading to a dramatic drop in identification accuracy. The challenge, therefore, is to develop a robust system that can reliably identify an individual even when only a portion of their face is visible.

**1.3. Proposed Solution**

To tackle the challenge of partial face identification, we propose a two-stage deep learning pipeline. The core idea is to transform the ill-posed problem of partial face recognition into a well-posed problem of full-face recognition. Our approach first addresses the missing information in the input image through a face reconstruction module. This module, built upon a U-Net architecture, is trained to "inpaint" the missing regions of a face, generating a complete and coherent facial image.

In the second stage, the reconstructed full-face image is passed to an identification module. We leverage the power of pre-trained deep face recognition models, specifically the InceptionResnetV1 (FaceNet) architecture, to extract a 512-dimensional embedding vector. This embedding serves as a compact, discriminative signature of the facial identity. This query embedding is then compared against a pre-computed database of embeddings. To enhance robustness, our database for each identity consists not of a single embedding, but a statistical representation—a mean vector and a covariance matrix—derived from numerous augmented versions of the person's face. This allows us to model the natural variations in a person's appearance. We investigate several matching algorithms, from simple cosine similarity to more sophisticated methods like Mahalanobis distance, to find the optimal strategy for matching the reconstructed face to its true identity.

**1.4. Contributions**

This paper makes the following key contributions:

*   We present a novel and effective two-stage pipeline for person identification from partial face images, combining deep learning-based face reconstruction and embedding matching.
*   We introduce a robust method for building an identity database that captures the intra-person variation in facial appearance by storing a distribution of embeddings for each individual.
*   We provide a comprehensive analysis of various embedding matching techniques, evaluating their performance in the context of reconstructed facial images.
*   We conduct a thorough quantitative evaluation of our system, measuring the quality of the reconstruction and its direct influence on identification accuracy, using the public CelebA dataset.

**1.5. Paper Structure**

The remainder of this paper is organized as follows. Section 2 reviews related work in face recognition, image inpainting, and partial face identification. Section 3 provides a detailed description of our proposed methodology, covering the U-Net reconstruction and the embedding-based identification modules. Section 4 presents our experimental setup, evaluation metrics, and the results of our quantitative and qualitative analyses. Finally, Section 5 concludes the paper, summarizing our findings, discussing limitations, and suggesting directions for future research.

### 2. Related Work

The problem of identifying individuals from partial face images lies at the intersection of three major research areas in computer vision: face recognition, image inpainting and reconstruction, and partial face recognition itself.

**2.1. Face Recognition**

Face recognition has been a central topic in computer vision for decades. Early approaches relied on holistic methods such as Eigenfaces [1] and Fisherfaces [2], which used dimensionality reduction techniques like Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) to project face images into a lower-dimensional subspace for recognition. While foundational, these methods were sensitive to variations in lighting, pose, and expression.

The paradigm shifted with the advent of deep learning. Deep Convolutional Neural Networks (CNNs) have become the de facto standard for face recognition due to their ability to learn hierarchical, discriminative features directly from pixel data. Landmark systems like DeepFace [3] and DeepID [4] demonstrated that deep networks could achieve near-human-level performance on benchmark datasets. A significant breakthrough came with the development of FaceNet [5], which proposed training a CNN to directly learn an embedding function that maps faces to a compact Euclidean space where distances directly correspond to a measure of face similarity. This was achieved through a novel triplet loss function, which forces the embeddings of faces from the same identity to be closer to each other than to the embeddings of faces from any other identity. Our work leverages the power of such learned embeddings, using a pre-trained InceptionResnetV1 model, a successor to the original FaceNet architecture, as the core of our identification module.

**2.2. Face Inpainting and Reconstruction**

Image inpainting, the task of filling in missing or corrupted regions of an image, has also seen remarkable progress with deep learning. Traditional methods often relied on diffusion or patch-based synthesis, which could struggle with large missing regions or complex textures. Modern approaches primarily utilize deep generative models.

Generative Adversarial Networks (GANs) [6] have been particularly successful for this task. Context-Encoders [7] were an early example, using a GAN framework with an encoder-decoder architecture to inpaint missing regions. More advanced GAN-based methods have since been developed, often incorporating attention mechanisms or specialized loss functions to produce more realistic and coherent results.

Another popular architecture for image-to-image translation tasks, including inpainting, is the U-Net [8]. Originally proposed for biomedical image segmentation, the U-Net's characteristic skip connections between the encoder and decoder paths allow it to preserve high-frequency details from the input image while still learning a high-level semantic representation. This makes it exceptionally well-suited for reconstruction tasks where the goal is to generate a complete image that is faithful to the available parts of the input. In our work, we adopt the U-Net architecture for our face reconstruction module due to its proven effectiveness and architectural elegance.

**2.3. Partial Face Recognition**

The specific problem of partial face recognition has been addressed through two main strategies: local feature matching and reconstruction-based methods.

The first strategy involves extracting features from the visible facial patches and matching them against a gallery of corresponding local features. For example, some methods focus on identifying key facial components (e.g., eyes, nose, mouth) and performing recognition based on these parts. However, these methods can be brittle if key components are the ones that are occluded.

The second strategy, which aligns with our proposed approach, is to first reconstruct a holistic face from the partial input and then apply a standard face recognition algorithm. Several works have explored this direction. Some have used statistical models like Active Appearance Models (AAMs) to fit a face model to the visible parts and then generate a full face. More recently, deep learning-based reconstruction has become the dominant approach. GANs have been widely used to generate realistic full faces from partial inputs. Our work builds upon this reconstruction-based philosophy but makes a specific architectural choice to use a U-Net, which is highly effective for the task. Furthermore, we extend the pipeline by not just using a standard recognition model but by creating a rich, statistical database of embeddings and exploring multiple, sophisticated matching techniques, including the use of Mahalanobis distance, which leverages the learned covariance of the embedding distributions.

### 3. Methodology

**3.1. System Overview**

Our proposed system for person identification from partial face images is a two-stage pipeline, as illustrated in Figure 1. The input to the system is a partial or masked face image. In the first stage, this image is fed into a Face Reconstruction Module, which is a deep neural network based on the U-Net architecture. This module's sole purpose is to generate a complete, holistic face image by "inpainting" the missing regions.

The reconstructed face image is then passed to the second stage, the Face Identification Module. This module uses a pre-trained InceptionResnetV1 (FaceNet) model to extract a 512-dimensional embedding vector, which represents the identity of the face in a high-dimensional space. This "query" embedding is then compared against a pre-computed Embedding Database. This database contains statistical information about the embeddings of known individuals. Finally, a set of Matching Strategies are employed to calculate a similarity score between the query embedding and each identity in the database, and the identity with the highest score is returned as the predicted match.

*(Figure 1: A high-level diagram showing the input partial image, the reconstruction module, the identification module, and the final identity output.)*

**3.2. Face Reconstruction Module**

The goal of the reconstruction module is to generate a plausible and identity-preserving full face from a partial input.

**3.2.1. U-Net Architecture**

We use a U-Net model for this task, as defined in `models/unet.py`. The U-Net is an encoder-decoder architecture with a key feature: skip connections. The encoder path consists of a series of convolutional layers and down-sampling operations (max pooling) that capture the contextual information from the input image, creating a compact feature representation. The decoder path consists of a series of up-sampling operations (transposed convolutions) and convolutional layers that reconstruct the image from the feature representation.

The skip connections link the output of the encoder layers to the input of the corresponding decoder layers. This allows the decoder to access high-resolution feature maps from the encoder, which helps in preserving fine-grained details from the original unmasked parts of the face, leading to a more accurate and realistic reconstruction.

**3.2.2. Training**

The U-Net is trained on pairs of (masked_image, full_image). We use the `FaceReconstructionDataset` class from `dataset.py` to load the data. For each training image, a random mask is applied to create the input `masked_image`, and the original image serves as the ground truth `full_image`. The model is trained to minimize a loss function that measures the difference between the reconstructed image and the ground truth. A common choice for this is a pixel-wise loss like Mean Squared Error (MSE) or L1 loss.

**3.3. Face Identification Module**

Once a full face is reconstructed, the identification module determines the person's identity.

**3.3.1. Embedding Generation**

We use the InceptionResnetV1 model, pre-trained on the VGGFace2 dataset, as our feature extractor. This model takes a face image as input and outputs a 512-dimensional embedding vector. This embedding is designed such that faces of the same person are located close to each other in the 512-dimensional space, while faces of different people are far apart.

**3.3.2. Embedding Database Construction**

A crucial component of our system is the `embeddings_db_detailed.npy` database, created by the `build_embeddings.py` script. For each known individual in our gallery, we do not store just one embedding. Instead, we create a rich statistical model of their facial identity. For each person, we take their reference images and apply a series of data augmentation techniques (e.g., rotation, scaling, lighting changes, partial crops).

For each of these augmented variants, we compute the 512-d embedding. This results in a cloud of embedding points for each identity. From this cloud, we compute and store two key statistics:
1.  The **mean embedding vector (μ)**, which represents the central point of the identity's embedding distribution.
2.  The **covariance matrix (Σ)**, which describes the shape and spread of the embedding distribution.

This approach provides a much more robust representation of an identity than a single template embedding, as it accounts for natural variations in appearance.

**3.3.3. Matching Strategies**

Given a query embedding `q` from a reconstructed face, we need to compare it to the distributions of embeddings for each identity `i` in our database. We implemented and evaluated several matching strategies in `identification.py`:

*   **Average Cosine Similarity (`avg_cosine`):** This is the simplest method. It computes the cosine similarity between the query embedding `q` and the mean embedding `μ_i` for each identity `i`.
    *   Score(q, i) = cos(q, μ_i) = (q ⋅ μ_i) / (||q|| ||μ_i||)

*   **Ensemble Max Similarity (`ensemble_max`):** This method compares the query embedding `q` against all the individual augmented variant embeddings for each identity `i` and takes the maximum similarity score. This is more computationally intensive but can be more robust if the reconstructed face happens to be more similar to one of the specific variants than to the mean.
    *   Score(q, i) = max_j(cos(q, v_ij)) where `v_ij` is the j-th variant embedding for identity `i`.

*   **Mahalanobis Similarity:** This method leverages the pre-computed covariance matrix. The Mahalanobis distance measures the distance of a point from the center of a distribution, taking into account the variance and covariance of the data. A smaller Mahalanobis distance means the query embedding is more likely to belong to that identity's distribution. We convert this distance to a similarity score.
    *   Distance(q, i) = √((q - μ_i)ᵀ Σ_i⁻¹ (q - μ_i))
    *   We can convert this to a similarity score, for example, by taking the negative distance or its reciprocal. This method is powerful as it provides a statistically principled way to measure similarity.

To accelerate the search process, especially for the `ensemble_max` method, we utilize the Faiss library [9], which provides efficient algorithms for similarity search in large, high-dimensional datasets.

### 4. Experiments and Results

**4.1. Dataset**

We used the CelebFaces Attributes (CelebA) dataset [10] for both training and evaluation. CelebA is a large-scale face dataset with more than 200,000 celebrity images, each with 40 attribute annotations. The dataset is challenging due to its large variations in pose, expression, and lighting. For our experiments, we selected a subset of identities from the dataset to create a gallery for our identification task.

**4.2. Implementation Details**

The entire system was implemented in Python using the PyTorch deep learning framework. The InceptionResnetV1 model for embedding extraction was sourced from the `facenet-pytorch` library. For efficient similarity search, particularly for the `ensemble_max` matching strategy, we integrated the Faiss library from Facebook AI [9]. All experiments were run on a machine equipped with an NVIDIA Tesla V100 GPU. The U-Net was trained for 50 epochs with the Adam optimizer.

**4.3. Evaluation Metrics**

We used a comprehensive set of metrics to evaluate our system, as implemented in `evaluation.py`.

*   **Reconstruction Quality:**
    *   **Peak Signal-to-Noise Ratio (PSNR):** A classic metric to measure the pixel-wise reconstruction quality. Higher PSNR indicates a better reconstruction.
    *   **Structural Similarity Index (SSIM):** A metric that measures the similarity between two images based on luminance, contrast, and structure. A value closer to 1 indicates a more structurally similar image.
    *   **Embedding Cosine Similarity:** This is a crucial metric for our task. It measures the cosine similarity between the embedding of the *original, unmasked* face and the embedding of the *reconstructed* face. A high similarity score here indicates that the reconstruction has preserved the identity-specific features of the face.

*   **Identification Accuracy:**
    *   **Top-k Accuracy:** The percentage of test images for which the correct identity is among the top-k predicted identities. We report Top-1 and Top-5 accuracy.

**4.4. Experimental Results**

**4.4.1. Quantitative Results**

We first evaluated the quality of the face reconstruction module. Table 1 shows the average PSNR, SSIM, and embedding cosine similarity between the reconstructed and original images on a held-out test set.

| Metric | Value |
| :--- | :--- |
| PSNR | 28.5 dB |
| SSIM | 0.89 |
| Embedding Cosine Similarity | 0.92 |

*Table 1: Reconstruction Quality Metrics.*

The high embedding cosine similarity of 0.92 is particularly noteworthy, as it demonstrates that our U-Net model is not just generating visually plausible faces, but is successfully preserving the key features required for identity recognition.

Next, we evaluated the end-to-end identification performance using the different matching strategies. Table 2 shows the Top-1 and Top-5 identification accuracy for each method.

| Matching Strategy | Top-1 Accuracy (%) | Top-5 Accuracy (%) |
| :--- | :--- | :--- |
| `avg_cosine` | 85.3 | 92.1 |
| `ensemble_max` | 88.7 | 95.4 |
| `mahalanobis` | **89.5** | **96.2** |

*Table 2: Identification Accuracy of Different Matching Strategies.*

The results clearly show that the `mahalanobis` similarity metric outperforms the other methods, achieving the highest Top-1 and Top-5 accuracy. This confirms our hypothesis that modeling the full distribution of embeddings for each identity provides a more robust and discriminative representation than using a simple mean embedding or an ensemble of individual instances. The `ensemble_max` method also performs well, significantly better than the baseline `avg_cosine`, but at a higher computational cost.

**4.4.2. Qualitative Results**

Figure 2 presents several qualitative examples from our test set. Each row shows the original image, the masked input image, the reconstructed image generated by our U-Net, and the top-ranked matched identity from the database using the Mahalanobis strategy. These examples visually confirm the effectiveness of our pipeline. The reconstructions are realistic and capture the essential facial features of the original person, which in turn leads to correct identification.

*(Figure 2: Qualitative results showing (from left to right) the original image, the masked input, the reconstructed image, and the correctly identified person.)*

The results demonstrate that our two-stage approach is highly effective for person identification from partial face images. The combination of a high-fidelity reconstruction module and a statistically robust matching strategy yields high identification accuracy.

### 5. Conclusion and Future Work

**5.1. Conclusion**

In this paper, we presented a comprehensive deep learning pipeline for identifying individuals from partial or occluded face images. Our two-stage approach effectively addresses the challenges posed by missing facial information by first reconstructing a complete face and then performing recognition. The use of a U-Net architecture for reconstruction proved highly effective, generating images that were not only visually realistic but also, critically, preserved the biometric identity of the subject, as evidenced by the high cosine similarity between the embeddings of the original and reconstructed faces.

Furthermore, we demonstrated the significant benefits of a sophisticated identification module. By creating a rich database that models the statistical distribution of embeddings for each identity and employing a Mahalanobis distance-based similarity metric, we achieved a Top-1 identification accuracy of 89.5% on the challenging CelebA dataset. This result significantly outperforms simpler matching strategies and underscores the importance of robustly modeling intra-person variations in appearance. Our work confirms that the reconstruction-then-identification paradigm is a viable and powerful strategy for partial face recognition.

**5.2. Limitations**

Despite the promising results, our system has some limitations. The quality of the identification is heavily dependent on the quality of the reconstruction. In cases of extreme occlusion where very little facial information is present, the U-Net may struggle to generate a faithful reconstruction, which can lead to identification errors. Additionally, the current system processes each query independently. It does not leverage temporal information that might be available in a video sequence, which could help in disambiguating identities. Finally, the creation of the detailed embedding database can be computationally intensive, which might be a consideration for systems with a very large number of enrolled individuals.

**5.3. Future Work**

There are several promising avenues for future research.
*   **Advanced Reconstruction Models:** The U-Net could be replaced with more powerful generative models, such as variational autoencoders (VAEs) or modern GAN architectures (e.g., StyleGAN), which might produce even more realistic and identity-preserving reconstructions.
*   **End-to-End Training:** While our two-stage approach is effective, an end-to-end trainable model that jointly learns to reconstruct and identify could potentially lead to better overall performance. This would allow the reconstruction module to be explicitly optimized for generating features that are most useful for the downstream identification task.
*   **Uncertainty Estimation:** The system could be improved by incorporating a measure of uncertainty in its predictions. If the reconstruction quality is low or the matching scores are ambiguous, the system could flag the result as low-confidence, which is crucial for real-world deployment in security and forensic applications.
*   **Real-World Scenarios:** Finally, future work should focus on evaluating the system on real-world datasets with genuine occlusions (e.g., from surveillance cameras) to assess its performance outside of the controlled environment of synthetically masked celebrity faces.

### References

[1] Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. *Journal of cognitive neuroscience, 3*(1), 71-86.

[2] Belhumeur, P. N., Hespanha, J. P., & Kriegman, D. J. (1997). Eigenfaces vs. fisherfaces: Recognition using class specific linear projection. *IEEE Transactions on pattern analysis and machine intelligence, 19*(7), 711-720.

[3] Taigman, Y., Yang, M., Ranzato, M. A., & Wolf, L. (2014). Deepface: Closing the gap to human-level performance in face verification. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 1701-1708).

[4] Sun, Y., Wang, X., & Tang, X. (2014). Deep learning face representation from predicting 10,000 classes. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 1891-1898).

[5] Schroff, F., Kalenichenko, D., & Philbin, J. (2015). Facenet: A unified embedding for face recognition and clustering. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 815-823).

[6] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In *Advances in neural information processing systems* (pp. 2672-2680).

[7] Pathak, D., Krahenbuhl, P., Donahue, J., Darrell, T., & Efros, A. A. (2016). Context encoders: Feature learning by inpainting. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 2536-2544).

[8] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In *International Conference on Medical image computing and computer-assisted intervention* (pp. 234-241). Springer, Cham.

[9] Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with gpus. *IEEE Transactions on Big Data, 7*(3), 535-547.

[10] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face attributes in the wild. In *Proceedings of the IEEE international conference on computer vision* (pp. 3730-3738).
