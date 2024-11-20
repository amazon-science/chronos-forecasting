# Chronos Forecasting with the Wasserstein Loss

## Overview

This repository is a **fork** of the original [Chronos Forecasting](https://github.com/amazon-science/chronos-forecasting/) project by Amazon Science. Below are the details of the original repository and the modifications made in this fork.

---

## Updates in This Fork

This fork was created to enhance and extend the functionality of the original repository. The installation process remains the same as in the original repository. The following changes and improvements have been made:

### 1. **Added the Wasserstein Loss**
   - The primary contribution of this fork is replacing the cross-entropy loss with the Wasserstein loss. See the related [research paper](https://arxiv.org/abs/2409.15367) for details.
   - Introduced a new binary variable in the training configuration: `use_wasserstein_loss`. Use this variable to specify whether to use the cross-entropy loss or the Wasserstein loss.
   - Added a new integer variable in the training configuration: `wasserstein_p`. Use this variable to specify which Wasserstein loss to apply (`W_1` or `W_2`).
   - To run the script, use the same command as in the original repository, but with the updated [config file](training/configs/chronos-t5-small-wasserstein.yaml):
     ```bash
     CUDA_VISIBLE_DEVICES=0 python training/train.py --config /path/to/modified/config.yaml \
         --model-id amazon/chronos-t5-small \
         --no-random-init \
         --max-steps 1000 \
         --learning-rate 0.001
     ```

---

### 2. **Added a Script to Extract Training Data from Zero-Shot Datasets**
   - The zero-shot datasets include both training and test data. 
   - This new script removes the test data and retains only the training data for further use.
   - Example usage:
     ```bash
     python zero_shot_datasets.py --save_dir zero_shot_datasets_train_val/ --config_path ../evaluation/configs/zero-shot.yaml
     ```

---

### 3. **Modified the Evaluation Script to Save Prediction Forecasts**
   - Added functionality to save predictions during the evaluation process for analysis.
   - Introduced three new variables in the evaluation script:
     - `return_forecast`: Enables the saving of forecasts.
     - `path_save_forecast`: Specifies the directory to save the forecasts.
     - `file_name_forecast`: Specifies the filename for the saved forecasts.
---

## Installation

Follow the [original installation instructions](https://github.com/amazon-science/chronos-forecasting#installation) for setting up the repository. 

---

## License

This repository follows the licensing terms of the original project.