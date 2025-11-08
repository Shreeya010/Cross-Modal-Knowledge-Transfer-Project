# Cross-Modal Knowledge Transfer for Learning Systemst
IITB EdTech Internship 2025 with DYPCET
Track 1 - Educational Data Analysis (EDA)

Project Overview:

This project addresses Problem ID-15: Cross-Modal Knowledge Transfer from the IITB EdTech Internship 2025. The objective is to train a model using EEG data as the teacher modality and evaluate if student models trained on eye-tracking (EYE + IVT), GSR, or facial expressions (TIVA) can approximate the teacher's performance through techniques like domain adaptation or modality dropout. Advanced methods include adversarial domain adaptation and contrastive learning.
The project uses multimodal physiological data to predict task accuracy (binary: Correct/Incorrect) or engagement levels from PSY.csv, focusing on synchronized per-trial features across modalities.

Key Objectives:

1.Use EEG to train a teacher model and test approximation by eye-tracking or GSR-only student models.

2.Implement knowledge distillation, adversarial domain adaptation, and contrastive learning.

3.Evaluate performance improvements and interpret key features (e.g., pupil size in eye-tracking).

The dataset consists of physiological signals from 38 participants across multiple modalities, organized per-trial for synchronization. Files are stored in the data/ directory:

*EEG.csv:

EEG signals (teacher modality). Features: Mean and variance of frequency bands (Delta, Theta, Alpha, Beta, Gamma).

*EYE.csv + IVT.csv:

Eye-tracking data (student modality). Features: Average fixation duration, saccade amplitude, mean pupil size.

*GSR.csv:

Galvanic Skin Response (student modality). Features: Mean conductance, slopes, recovery rates.

*TIVA.csv:

Facial expressions (student modality). Features: Average Action Unit (AU) intensities or emotion probabilities.

*PSY.csv:

Psychological data for targets (e.g., Engagement levels or task accuracy).

Data Organization:

Features are extracted and paired per trial, ensuring teacher-student alignment (e.g., EEG features matched with Eye features from the same event).

#Note: Due to data access issues, only EEG features were fully processed in this implementation. Student modalities (Eye, GSR, Facial) are placeholders for future integration.

Installation & Setup

Clone the Repository:

textgit clone <your-repo-url>

cd project

Environment Setup (Google Colab recommended):

Mount Google Drive:

from google.colab import drive; drive.mount('/content/drive')

Install dependencies:

text!pip install pandas numpy scikit-learn xgboost tensorflow shap matplotlib seaborn

Data Preparation:

Place dataset files in data/ or mount /content/drive/MyDrive/STData/STData/ for participant subdirectories (1-38).

Ensure PSY.csv is available for target encoding.

Usage:

Run the notebooks sequentially in Jupyter/Colab:

01_preprocessing.ipynb:

Load data, extract features (EEG frequency bands, Eye summaries, GSR slopes, TIVA AUs), normalize with z-score, apply PCA (10 components per modality), and save preprocessed_trials.csv.

02_baseline_single_modality.ipynb:

Train XGBoost baselines on each modality and evaluate F1-score, Accuracy, ROC-AUC.

03_teacher_student_transfer.ipynb:

Train EEG teacher, then distill knowledge to student models (Eye, GSR, Facial) using soft labels.

04_domain_adaptation.ipynb:

Implement adversarial adaptation with gradient reversal for domain-invariant features.

05_contrastive_learning.ipynb:

Use NT-Xent loss for shared embedding space across modalities.

Example Command (Colab):

Open each notebook and run all cells. Outputs include saved models, metrics CSVs, and plots.

Key Hyperparameters:

1.XGBoost: random_state=42, scale_pos_weight for imbalance.

2.Distillation: Temperature=2.0, Alpha=0.5.

3.Adaptation: Embedding dim=16, Loss weights=[1.0, 0.5, 0.5].

#Results

Baseline EEG Performance

MetricValue

F1-score0.5263

Accuracy0.6213

ROC-AUC0.7595

Future Work

1.Resolve Step 2 data loading for full modalities.

2.Implement hyperparameter tuning (GridSearchCV) and 5-fold cross-validation.

3.Explore modality dropout and pretraining for robustness.

4.Ensemble models across modalities.

5.Full SHAP analysis for student models post-transfer
