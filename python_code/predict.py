from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr


def _posterior_predictive_draw_per_row(
    model,  # bambi.Model
    idata,  # arviz.InferenceData
    new_data_pd: pd.DataFrame,
    *,
    rng: np.random.Generator,
    include_group_specific: bool = True,
    sample_new_groups: bool = False,
    random_seed: Optional[int] = None,
    # NEW: memory control
    chunk_size: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DROP-IN replacement (same signature + returns) that avoids allocating
    (N, chains, draws) arrays for the full dataset.

    Strategy:
      - Process new_data in chunks
      - For each chunk, select ONE posterior draw (one chain+draw) and call
        model.predict() on that reduced idata (chain=1, draw=1).
      - Return one posterior predictive draw per row, plus an integer draw_id.

    Notes:
      - draw_id is an integer index into the *flattened* (chain, draw) space:
            draw_id = chain_idx * n_draws + draw_idx
      - This is not "independent draw per row" across all N rows; it's "one draw
        per chunk". It’s the only practical way to keep memory bounded at 1.7M rows.
      - Increase/decrease chunk_size to balance speed vs RAM.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    n_chain = int(idata.posterior.sizes["chain"])
    n_draw = int(idata.posterior.sizes["draw"])

    ys: List[np.ndarray] = []
    draw_ids: List[np.ndarray] = []

    N = len(new_data_pd)

    # Iterate over pandas slices to avoid copying the whole table repeatedly
    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)
        chunk_pd = new_data_pd.iloc[start:stop]

        # Pick one posterior draw for this chunk
        c = int(rng.integers(0, n_chain))
        d = int(rng.integers(0, n_draw))
        flat_draw_id = c * n_draw + d

        # Reduce idata to a single draw to keep predict() tiny in memory
        idata_1 = idata.isel(chain=[c], draw=[d])

        pred_idata = model.predict(
            idata_1,
            kind="response",
            data=chunk_pd,
            inplace=False,
            include_group_specific=include_group_specific,
            sample_new_groups=sample_new_groups,
            random_seed=random_seed,
        )

        pp_group = pred_idata.posterior_predictive
        var_name = list(pp_group.data_vars)[0]
        da: xr.DataArray = pp_group[var_name]

        # Find the obs dimension name (not chain/draw)
        obs_dim = [dim for dim in da.dims if dim not in ("chain", "draw")][0]

        # For chain=1, draw=1 this is safe and small
        # shape typically (1, 1, n_obs_chunk) -> flatten to (n_obs_chunk,)
        y_chunk = np.asarray(da.transpose("chain", "draw", obs_dim).values).reshape(-1)

        ys.append(y_chunk)
        draw_ids.append(np.full((stop - start,), flat_draw_id, dtype=np.int32))

    return np.concatenate(ys, axis=0), np.concatenate(draw_ids, axis=0)


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
    # NEW: control memory/speed tradeoff
    chunk_size: int = 50_000,
) -> pl.DataFrame:
    """
    DROP-IN replacement for your current predict_allmodels() that avoids the
    50+ GiB allocation by chunking + selecting a single posterior draw per chunk.

    Output columns match your version:
      any_asset, asset_pred, asset_draw, any_debt, debt_pred, debt_draw
    """
    rng = np.random.default_rng(seed)

    if ndraw < 1:
        raise ValueError("ndraw must be >= 1")

    # Convert once for Bambi
    base_pd = data.to_pandas()

    out_frames = []

    for k in range(ndraw):
        block_seed = None if seed is None else (seed + k)

        any_asset, _ = _posterior_predictive_draw_per_row(
            asset_class_model,
            asset_class_idata,
            base_pd,
            rng=rng,
            include_group_specific=include_group_specific,
            sample_new_groups=sample_new_groups,
            random_seed=block_seed,
            chunk_size=chunk_size,
        )

        asset_pred, asset_draw = _posterior_predictive_draw_per_row(
            asset_model,
            asset_idata,
            base_pd,
            rng=rng,
            include_group_specific=include_group_specific,
            sample_new_groups=sample_new_groups,
            random_seed=block_seed,
            chunk_size=chunk_size,
        )

        any_debt, _ = _posterior_predictive_draw_per_row(
            debt_class_model,
            debt_class_idata,
            base_pd,
            rng=rng,
            include_group_specific=include_group_specific,
            sample_new_groups=sample_new_groups,
            random_seed=block_seed,
            chunk_size=chunk_size,
        )

        debt_pred, debt_draw = _posterior_predictive_draw_per_row(
            debt_model,
            debt_idata,
            base_pd,
            rng=rng,
            include_group_specific=include_group_specific,
            sample_new_groups=sample_new_groups,
            random_seed=block_seed,
            chunk_size=chunk_size,
        )

        # Match your R semantics: treat any_asset/any_debt as 0/1 and zero-out preds when class=0
        any_asset_i = any_asset.astype(np.int8)
        any_debt_i = any_debt.astype(np.int8)

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

        if ndraw > 1:
            df_k = df_k.with_columns(pl.lit(k).alias("draw_block").cast(pl.Int32))

        out_frames.append(df_k)

    return pl.concat(out_frames, how="vertical") if ndraw > 1 else out_frames[0]