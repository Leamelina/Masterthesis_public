# Master Thesis Project  
**Title:** *Analysing a Surrogate Neural Network Model to Quantify Controlling Factors of a Global Hydrological Model*

## Overview

This project is part of a master's thesis focused on building and analyzing a **surrogate neural network model** for a **Global Hydrological Model (GHM)**. The surrogate model is designed to approximate the behavior of the GHM, enabling more efficient application of **explainable AI (XAI)** and **sensitivity analysis** methods.

The primary objective is to identify and quantify the influence of various input variables on hydrological outputs across different climate regions.

## Approach

- A neural network was trained as a **surrogate model** to emulate the behavior of a computationally expensive GHM.
- The surrogate model was used to apply XAI and sensitivity analysis techniques that would be otherwise computationally infeasible on the full model.
- The input data was stratified into **four climatic regions**:
  - Wet & Warm  
  - Wet & Cold  
  - Dry & Warm  
  - Dry & Cold  
- Analysis was conducted both **globally** and **regionally** to compare the dominant controlling factors in different climates.

## Methods Used

- **LassoNet** – Feature selection using structured sparsity.
- **SHAP (SHapley Additive exPlanations)** – Explanation of model outputs based on feature contributions.
- **Variance-Based Sensitivity Analysis (VBSA)** – Quantifies output sensitivity to input variables.
- All methods were **bootstrapped** to estimate **uncertainty** in the results.

### Additional Notes

- **Multipass Permutation Variable Importance (MPVI)** was tested but **discarded** due to unsatisfactory performance.
- All methods were validated on synthetic datasets before being applied to the full surrogate model.

## Purpose

By analyzing the surrogate model with these methods, this project aims to:

- Improve understanding of how input variables affect GHM outputs.
- Identify regional differences in variable importance.
- Provide a framework for efficient and interpretable analysis of large-scale hydrological models.

## Dependencies and Licensing  

This project uses the **LassoNet** library, developed by **Louis Abraham and Ismael Lemhadri**, which is released under the **MIT License**.  

- Official repository: [https://github.com/lasso-net/lassonet](https://github.com/lasso-net/lassonet)  
- Copyright © 2020 Louis Abraham, Ismael Lemhadri  

LassoNet is used in this project for subset selection and feature importance ranking.  

This project also uses the **PermutationImportance** library, developed by **gelijergensen**, which is released under the **MIT License**.  

- Official repository: [https://github.com/gelijergensen/PermutationImportance](https://github.com/gelijergensen/PermutationImportance)  
- Copyright © 2019 gelijergensen  

PermutationImportance is used in this project for testing methods for measuring importance on synthetic data.

