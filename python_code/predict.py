from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr


def _posterior_predictive_draw_per_row(
    model,  # bambi.Model
    idata,  # arviz.InferenceData from model.fit()
    new_data_pd: pd.DataFrame,
    *,
    rng: np.random.Generator,
    include_group_specific: bool = True,
    sample_new_groups: bool = False,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (y_draw, draw_id) where:
      - y_draw is one posterior predictive draw per observation (row)
      - draw_id is the sampled posterior draw index per observation

    Note: This intentionally samples draws independently per row (like your expanded+rowwise setup).
    """
    # Ask Bambi for posterior predictive samples for *all* posterior draws
    # Bambi API: Model.predict(idata, kind="response", data=..., inplace=..., include_group_specific=..., sample_new_groups=...)
    pred_idata = model.predict(
        idata,
        kind="response",
        data=new_data_pd,
        inplace=False,
        include_group_specific=include_group_specific,
        sample_new_groups=sample_new_groups,
        random_seed=random_seed,
    )

    # Grab the single posterior predictive variable (usually the response name)
    pp_group = pred_idata.posterior_predictive
    var_name = list(pp_group.data_vars)[0]
    da: xr.DataArray = pp_group[var_name]  # dims: (chain, draw, __obs__ or similar)

    # Stack chain/draw into one dimension => (sample, obs)
    # obs dim name can differ; find it:
    obs_dim = [d for d in da.dims if d not in ("chain", "draw")][0]
    pp = da.stack(sample=("chain", "draw")).transpose("sample", obs_dim)

    arr = np.asarray(pp.values)  # shape: (S, N)
    S, N = arr.shape

    # For each row, sample a posterior-draw index independently
    draw_ids = rng.integers(0, S, size=N, endpoint=False)

    # Pull one predictive draw per row
    y_draw = arr[draw_ids, np.arange(N)]

    return y_draw, draw_ids


def predict_allmodels(
    data: pl.DataFrame,
    *,
    asset_class_model,
    asset_class_idata,
    asset_model,
    asset_idata,
    debt_class_model,
    debt_class_idata,
    debt_model,
    debt_idata,
    ndraw: int = 1,
    seed: Optional[int] = None,
    include_group_specific: bool = True,
    sample_new_groups: bool = False,
) -> pl.DataFrame:
    """
    Python analogue of your R predict_allmodels():
      - any_asset: posterior predictive draw from asset_class_model
      - asset_pred: posterior predictive draw from asset_model (forced to 0 if any_asset==0)
      - any_debt: posterior predictive draw from debt_class_model
      - debt_pred: posterior predictive draw from debt_model (forced to 0 if any_debt==0)
      - asset_draw / debt_draw: draw index used per row (similar role to .draw in your R output)

    IMPORTANT:
      This returns ONE row per input row (like ndraw=1).
      If you set ndraw > 1, it repeats the dataset ndraw times (like tidybayes).
    """
    rng = np.random.default_rng(seed)

    # Convert to pandas for Bambi's predict() API
    base_pd = data.to_pandas()

    if ndraw < 1:
        raise ValueError("ndraw must be >= 1")

    out_frames = []

    for k in range(ndraw):
        # Independent RNG stream per draw “block”
        # (matches tidybayes idea of multiple draws; but still independent-per-row within each block)
        block_seed = None if seed is None else (seed + k)

        any_asset, _ = _posterior_predictive_draw_per_row(
            asset_class_model,
            asset_class_idata,
            base_pd,
            rng=rng,
            include_group_specific=include_group_specific,
            sample_new_groups=sample_new_groups,
            random_seed=block_seed,
        )

        asset_pred, asset_draw = _posterior_predictive_draw_per_row(
            asset_model,
            asset_idata,
            base_pd,
            rng=rng,
            include_group_specific=include_group_specific,
            sample_new_groups=sample_new_groups,
            random_seed=block_seed,
        )

        any_debt, _ = _posterior_predictive_draw_per_row(
            debt_class_model,
            debt_class_idata,
            base_pd,
            rng=rng,
            include_group_specific=include_group_specific,
            sample_new_groups=sample_new_groups,
            random_seed=block_seed,
        )

        debt_pred, debt_draw = _posterior_predictive_draw_per_row(
            debt_model,
            debt_idata,
            base_pd,
            rng=rng,
            include_group_specific=include_group_specific,
            sample_new_groups=sample_new_groups,
            random_seed=block_seed,
        )

        # Force zeros like your case_when()
        # (treat any_asset/any_debt as 0/1 draws)
        any_asset_i = any_asset.astype(int)
        any_debt_i = any_debt.astype(int)

        asset_pred = np.where(any_asset_i == 0, 0.0, asset_pred)
        debt_pred = np.where(any_debt_i == 0, 0.0, debt_pred)

        df_k = data.with_columns(
            pl.Series("any_asset", any_asset_i).cast(pl.Int8),
            pl.Series("asset_pred", asset_pred).cast(pl.Float64),
            pl.Series("asset_draw", asset_draw).cast(pl.Int32),
            pl.Series("any_debt", any_debt_i).cast(pl.Int8),
            pl.Series("debt_pred", debt_pred).cast(pl.Float64),
            pl.Series("debt_draw", debt_draw).cast(pl.Int32),
        )

        # If ndraw > 1, tag which draw-block this is (optional but helpful)
        if ndraw > 1:
            df_k = df_k.with_columns(pl.lit(k).alias("draw_block").cast(pl.Int32))

        out_frames.append(df_k)

    return pl.concat(out_frames, how="vertical") if ndraw > 1 else out_frames[0]