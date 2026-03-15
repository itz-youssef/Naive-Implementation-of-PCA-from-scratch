4.1 Dataset Selection & Description
two distinct datasets were selected to evaluate the performance of Naive Bayes and reduction techniques on different data types.

Dataset 1: Categorical Data
Name: Mushroom Classification Dataset.
Source: Kaggle.
Justification: This dataset is ideal for categorical classification because all of its 22 features (e.g., cap-shape, odor, spore-print-color) are discrete categories.
Size: 8,124 instances.
Number of Features: 22 categorical features.
Class Distribution: 4,208 Edible (51.8%) vs. 3,916 Poisonous (48.2%).
Dataset 2: Numerical Data
Name: Breast Cancer Wisconsin (Diagnostic) Dataset.
Source: Scikit-learn built-in datasets
Justification: This dataset is perfectly suited for numerical classification as it contains continuous computed features (e.g., mean radius, mean texture, mean smoothness) from digitized images of fine needle aspirates (FNA) of breast masses.
Size: 569 instances.
Number of Features: 30 continuous numerical features.
Class Distribution: 357 Benign (62.7%) vs. 212 Malignant (37.3%).
4.2 Implementation Details: PCA From Scratch
The Principal Component Analysis (PCA) algorithm was implemented from scratch using fundamental NumPy operations. The implementation follows these key mathematical steps:

Standardization (Mean Centering & Scaling): Before applying PCA, the data must be scaled so that each feature contributes equally. We calculate the mean  
μ
  and standard deviation  
σ
  for each feature.
Z
=
X
−
μ
σ
 
Purpose: Ensures that features with larger scales do not dominate the variance calculations.

Covariance Matrix Calculation: We compute the covariance matrix to understand how the variables relate to one another and vary together.
C
=
1
n
−
1
Z
T
Z
 
Purpose: Captures the linear relationships (correlations) between all pairs of features.

Eigendecomposition: We calculate the eigenvalues  
λ
  and eigenvectors  
v
  of the covariance matrix using np.linalg.eigh.
C
v
=
λ
v
 
Purpose: Eigenvectors represent the directions of maximum variance (Principal Components), and eigenvalues represent the magnitude of variance in those directions. The eigenvectors are then sorted in descending order based on their corresponding eigenvalues.

Projection (Dimensionality Reduction): We select the top  
k
  eigenvectors (forming a projection matrix  
W
 ) and project the original standardized data onto this new lower-dimensional space.
X
p
c
a
=
Z
⋅
W
 
Purpose: Transforms the dataset into a new space with reduced dimensions while retaining the maximum possible variance.

4.3 Results and Comparison
Comparison Table
Experiment	Dataset 1: Mushroom (Categorical)	Dataset 2: Breast Cancer (Numerical)
Exp 0: Baseline (All Features)	94.59%	94.15%
Exp A: Feature Selection	94.30%	94.15%
Exp B: PCA (From Scratch)	94.75%	92.98%
Questions
1. How did Naive Bayes perform on categorical versus numerical data? Naive Bayes generally performs exceptionally well on the categorical data (Mushroom dataset) using CategoricalNB, often achieving near-perfect accuracy. This is because the features have very strong, distinct conditional probabilities for each class. On the numerical data (Breast Cancer) using GaussianNB, it also performs very well but slightly lower than the categorical, as it assumes the numerical features follow a strict normal (Gaussian) distribution, which might not be perfectly true for all features.

2. Which approach achieved better results for each dataset? Why?

For the Numerical Dataset: Feature Selection or Baseline usually yields the best accuracy. PCA also performs well but might drop a slight percentage of accuracy due to information loss during dimensionality reduction.
For the Categorical Dataset: Baseline and PCA perform best.
3. How did both methods compare to the baseline model?

Feature Selection: Often matches or very slightly increasse compared to the baseline accuracy, offers the advantage of a simpler, faster model with fewer features.
PCA: Slightly lower accuracy than the baseline (since variance is reduced), but it significantly reduces dimensionality while maintaining highly competitive predictive power, especially for the numerical dataset.
4. What are the trade-offs between feature selection and feature reduction?

Feature Selection: Keeps original features, making the model highly interpretable (we know exactly which features are important). However, it completely discards the less selected features, potentially losing some hidden patterns.
Feature Reduction (PCA): Combines features into new Principal Components. It retains the core variance from all features, making it great for handling multicollinearity. The trade-off is a complete loss of interpretability—the new components are mathematical constructs, not real-world features.
5. Is PCA appropriate for categorical data? Discuss any issues observed. Theoretically, PCA is NOT appropriate for categorical data. PCA relies on variance, Euclidean distances, and linear relationships. When categorical variables are encoded into numbers, these numbers represent distinct categories, not mathematical magnitudes. Therefore, calculating the "mean" or "variance" of categories like "red", "blue", and "green" is mathematically meaningless. While the code will execute it, the resulting principal components lose their logical meaning, and the model's accuracy often drops compared to using standard categorical methods.
