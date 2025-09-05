import numpy as np


def prepare_covariates(entry: dict) -> dict:

    past_covariates = entry.get("past_feat_dynamic_real", None)
    if past_covariates is not None:
        #  missing value imputation for covariates
        past_covariates[np.isnan(past_covariates)] = 0

        # normalize covariates by mean of absolute values,
        covariates_scale = np.mean(np.abs(past_covariates), axis=0)
        covariates_scale[covariates_scale < 1] = 1.0
        past_covariates = past_covariates / covariates_scale

        # shift covariates by one
        num_covariates = past_covariates.shape[-1]
        past_covariates = np.concatenate(
            [past_covariates, np.array([[0] * num_covariates])], axis=0
        )

    future_covariates = entry.get("future_feat_dynamic_real", None)
    if future_covariates is not None:
        #  missing value imputation for covariates
        future_covariates[np.isnan(future_covariates)] = 0

        # normalize covariates by mean of absolute values,
        future_covariates = future_covariates / covariates_scale

        num_covariates = future_covariates.shape[-1]
        future_covariates = np.concatenate(
            [future_covariates, np.array([[0] * num_covariates])], axis=0
        )

    return {
        "past_covariates": past_covariates.astype(np.float32),
        "future_covariates": future_covariates.astype(np.float32),
    }
