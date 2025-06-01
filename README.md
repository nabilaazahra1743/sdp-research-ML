# 🧠 Software Defect Prediction Using CK Metrics with Ensemble Stacking

This research explores the potential of **AI-driven software defect prediction** by utilizing **CK (Chidamber & Kemerer) object-oriented metrics** and applying **ensemble learning techniques** to improve classification accuracy. The project uses data from open-source Java projects and applies a multi-level machine learning approach to identify buggy software components.


## 📄 Abstract Summary

Compared to manual and automated testing, AI-driven testing offers a more intelligent approach, enabling earlier **prediction of software defects** and improving testing efficiency.

This project aims to predict software defects using classification models based on **CK metrics** collected from **five open-source Java repositories on GitHub**. A total of **8924 data points** were processed, and after **data cleaning, normalization**, and **undersampling** (to resolve class imbalance), a final dataset of **1314 instances** was prepared:

- ✅ 746 non-defective (clean)
- ❌ 568 defective (buggy)

## 📊 Methodology

### 1. **Data Collection**
- Source: 5 open-source GitHub Java repositories
- Metric: CK metrics (e.g., WMC, DIT, NOC, CBO, LCOM, etc.)

### 2. **Preprocessing**
| Step              | Description                                         |
|------------------|-----------------------------------------------------|
| Cleaning          | Removed irrelevant or incomplete records           |
| Normalization     | Feature scaling applied to numerical metrics        |
| Undersampling     | Applied to handle class imbalance                  |

### 3. **Modeling**
The predictive model is developed in **two stages**:

#### 🔹 Base Learners (Level-0)
Tree-based ensemble classifiers:
- AdaBoost
- Random Forest (RF)
- Extra Trees (ET)
- Gradient Boosting (GB)
- Histogram-based Gradient Boosting (HGB)
- XGBoost (XGB)
- CatBoost (CAT)

#### 🔹 Meta Learner (Level-1)
- **Stacking ensemble** that combines base learners to optimize performance. The  meta learner using RandomForest as a final predictors. 

## 📈 Results

- ✅ **Stacking Model ROC-AUC Score**: `0.8575`
- 📉 Outperformed all individual classifiers
- 📌 **Best confusion matrix** (high TP/TN, low FP/FN)
- 📊 **Paired t-test** used for validation (all p-values < 0.05)
  - Best gain vs Gradient Boosting: `+0.0411` (p = `0.0030`)

These results confirm that the **stacking ensemble** offers a significantly more robust and balanced classifier for software defect prediction.


## 📁 Project Structure

| File/Folder                                                      | Description                                                               |
|------------------------------------------------------------------|---------------------------------------------------------------------------|
| `venv`                                                           | Virtual environment to manage project dependencies                        |
| `Dataset/`                                                       | Contains fixed undersampling datasets                                     |
| `Models/`                                                        | Saved trained models (if exported)                                        |
| `archieve-dataset/`                                              | Archieve dataset contained raw data, imbalanced techniques                |
| `catboost_info`                                                  | Auto-generated folder from CatBoost during model training                 |
| `statistical`                                                    | Contains notebooks related to statistical analysis, like paired t-tests   |
| `v0_sdp_prediction_intial.ipynb`                                 | Initial exploration and defect prediction using CK metrics                |
| `v1_sdp_overfitting.ipynb`                                       | Early analysis of overfitting risks in base models                        |
| `v2_sdp_preds_stacking_new_dataset.ipynb`                        | Implements stacking ensemble on the newly preprocessed dataset            |
| `v3_sdp_preds_undersampling.ipynb`                               | Model evaluation after applying undersampling to handle class imbalance   |
| `v4_sdp_for_for_test.ipynb`                                      | General testing notebook for the defect prediction pipeline               |
| `v5_sdp_preds_each base model_undersampling.ipynb`               | Evaluates each base model after undersampling                             |
| `v6_sdp_each base model_.ipynb`                                  | Baseline evaluation of each individual base model                         |
| `v7_sdp_undersampling_without_parameter.ipynb`                   | Undersampling experiment without any parameter tuning                     |
| `v8_sdp_undersampling_crossval_parameter adjust.ipynb`           | Combines undersampling with cross-validation and parameter adjustment     |
| `v9_sdp_undersampling_parameter fixed.ipynb`                     | Uses finalized hyperparameters after tuning                               |
| `v10_sdp_adjustable effective code.ipynb`                        | A more streamlined and adjustable version of the training code            |
| `v11_sdp_undersampling_create model pkl.ipynb`                   | Creates and saves the final trained model as a .pkl file                  |
| `v12_undersmpling_hyperparameter trial_optuna copy 2.ipynb`      | Hyperparameter tuning experiment using Optuna framework                   |
| `v13_sdp_undersampling_optimized parameter n create pkl.ipynb`   | Final stacking model with optimized parameters and model export           |
| `v14_testing_with_dataval.ipynb`                                 | Model testing using a separate validation dataset                         |
| `v15_undersmpling_sync_hiperparameter.ipynb`                     | Synchronizes hyperparameters across models to balance performance         |
| `v16_fixed_parameter.ipynb`                                      | Final notebook with fixed parameters used for evaluation and testing      |
| `readme.MD`                                                      | Project documentation (this file)                                         |


## ⚙️ Requirements

- Python any version
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - catboost
  - matplotlib / seaborn
  - imbalanced-learn
 
## 📊 Evaluation Metric

- ROC-AUC
- Confusion Matrix (TP, FP, TN, FN)
- Paired t-test for statistical significance


## 📌 Key Contributions

- Developed a robust software defect prediction model using tree-based ensemble methods.
- Demonstrated that ensemble stacking significantly improves defect prediction performance.
- Validated statistical significance of performance gains using hypothesis testing.
- Published the work in an indexed scientific journal.


## ✨ Citation

If you find this work helpful, consider citing the following publication:
@article{zahra2025sdp,
title={Software Defect Prediction using CK Metrics and Ensemble Stacking},
author={Zahra, Nabila A.},
journal={International Journal of Applied Data and Information Sciences (IJADIS)},
volume={5},
number={2},
year={2024},
url={https://ijadis.org/index.php/ijadis/article/view/1368}
}
