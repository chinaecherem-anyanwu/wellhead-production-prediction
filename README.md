# Oil Production Prediction from Wellhead Data Using Machine Learning

Predicting daily oil production (BPD) from wellhead sensor data using machine learning. Built on 2,196 observations from 22 Nigerian oil wells, comparing 6 algorithms across 5 engineered feature sets. Best model (CatBoost) achieved R² = 0.903, with water cut, choke size, and pressure identified as the dominant physical drivers of production.


## Overview

For decades, oilfield engineers have relied on empirical correlations like Gilbert's equation (1954) to estimate oil production from surface measurements. These formulas were built on a small, decades-old dataset from California and don't hold up well against the non-linear behavior of modern multiphase flow — especially in fields far from where they were originally calibrated, like Nigeria's.

This project asks whether machine learning can do better: can a model predict daily oil output directly from wellhead sensor readings, without relying on a hand-derived physics equation?


## Dataset

- **2,196 observations** from **22 Nigerian oil wells**
- After removing null/zero-production entries (shut-in wells): **2,193 observations**
- 75/25 train-test split (1,644 / 549 samples), fixed seed for reproducibility


## Methodology

The project follows a five-phase data science workflow:

1. **Data Acquisition & Integrity** — loading, cleaning, removing invalid entries
2. **Exploratory Data Analysis & Preprocessing** — correlation analysis, PCA visualization, median imputation, standardization
3. **Feature Engineering** — five feature sets tested:
   - Raw (19 features)
   - PCA (13 components, 95.46% variance explained)
   - Kernel PCA (8 components, RBF kernel)
   - Top-6 features by Random Forest importance
   - Top-10 features by XGBoost importance
4. **Model Selection & Optimization** — 6 algorithms (KNN, SVM-RBF, Decision Tree, Random Forest, XGBoost, CatBoost) evaluated across all 5 feature sets using 10-fold cross-validation; top 8 configurations shortlisted and tuned via 5×5 nested cross-validation with grid search
5. **Final Evaluation & Diagnostics** — retraining on full training set, validation on held-out test set, parity/residual plots


## Results

The best-performing model — **CatBoost trained on the top 10 XGBoost-selected features** — explains over 90% of the variance in oil production using only surface wellhead measurements.

**Key drivers identified** (consistent across Random Forest and XGBoost importance rankings): water cut (`bsw_pct`), choke size (`choke_64`), and wellhead pressure (`fthp_psig`) — aligning with established petroleum engineering process physics rather than reflecting spurious correlations.

Simpler models (Decision Tree, SVM) were excluded after diagnostic learning curves showed clear overfitting and underfitting respectively, confirming the non-linear nature of the production system.


## Repository Contents

```
├── Anyanwu_CHE_696_Final_Project_Code.ipynb   # Full analysis notebook
├── Anyanwu_CHE_696_Project_Report.pdf         # Full written report
├── CHE_696_Final_Project_Slides.pptx          # Presentation slides
└── README.md
```


## Tech Stack

- Python, Pandas, NumPy
- scikit-learn (KNN, SVM, Decision Tree, Random Forest, PCA, KPCA, GridSearchCV)
- XGBoost, CatBoost
- Matplotlib / Seaborn (visualization)


## Future Work

- Expand the dataset to leverage non-plateaued learning curves and further improve model precision
- Extend well-to-well analysis across a broader range of Nigerian oil fields
- Explore deployment as a lightweight tool for estimating production on wells with little or no production history


## References

1. Z. Xun et al., "Developing a cost-effective tool for choke flow rate prediction in sub-critical oil wells using wellhead data," *Sci. Rep.*, vol. 15, no. 1, Dec. 2025.
2. J. O. Olaide, O. A. Adeyanju, A. B. Ehinmowo, "Machine Learning Models for Predicting Flowrate for Niger Delta Oil Wells," Unpublished, Jun. 2023.


## Author

**Chinaecherem Anyanwu** — CHE 696 (University of Michigan) Final Project, April 2026
