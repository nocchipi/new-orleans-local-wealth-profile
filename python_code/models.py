import bambi as bmb
import polars as pl
import numpy as np


def create_models(sipp_model):
    
    any_asset_model_cols = ['hh_any_asset',
                            'state',
                            'hh_income',
                            'age',
                            'race_eth',
                            'edu',
                            'tenure',
                            'household_type',
                            'male',
                            'metro',
                            'disability',
                            'class_worker',
                            'public_assistance',
                            'social_security',
                            'poverty',
                            'citizen']
    
    any_asset_model_df = sipp_model.select(pl.col(any_asset_model_cols)).to_pandas().dropna(subset=any_asset_model_cols)
    
    for col in ['state','hh_income','age','race_eth','edu','tenure','household_type','class_worker']:
        any_asset_model_df[col] = any_asset_model_df[col].astype('category')

    any_asset_model_formula = """
                                hh_any_asset ~ male + metro + disability + class_worker + public_assistance + social_security +
                                poverty + citizen + tenure + (1|state) + (1|hh_income) + (1|age) + (1 | race_eth) + 
                                (1 | edu) + (1|household_type)
                                """
    
    any_asset_priors = {
            "Intercept": bmb.Prior("Normal", mu=0, sigma=2.5),
            "common": bmb.Prior("Normal", mu=0, sigma=1),
            "group_specific": bmb.Prior(
                "Normal",
                mu=0,
                sigma=bmb.Prior("Exponential", lam=1),
            )
        }
    
    any_asset_model = bmb.Model(formula=any_asset_model_formula, 
                            data=any_asset_model_df, 
                            family="bernoulli",
                            priors=any_asset_priors,
                            noncentered=True)
    
    
    any_debt_model_cols = ['hh_any_debt',
                        'state',
                        'hh_income',
                        'age',
                        'race_eth',
                        'edu',
                        'tenure',
                        'household_type',
                        'male',
                        'metro',
                        'disability',
                        'class_worker',
                        'public_assistance',
                        'social_security',
                        'poverty',
                        'citizen']
    
    any_debt_model_df = sipp_model.select(pl.col(any_debt_model_cols)).to_pandas().dropna(subset=any_debt_model_cols)
    
    for col in ['state','hh_income','age','race_eth','edu','tenure','household_type','class_worker']:
        any_debt_model_df[col] = any_debt_model_df[col].astype('category')
        
    any_debt_model_formula = """
    hh_any_debt ~ male + metro + disability + class_worker + public_assistance + social_security +
    poverty + citizen + tenure + (1|state) + (1|hh_income) + (1|age) + (1 | race_eth) + 
    (1 | edu) + (1|household_type)
    """
    
    any_debt_priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=2.5),
        "common": bmb.Prior("Normal", mu=0, sigma=1),
        "group_specific": bmb.Prior(
            "Normal",
            mu=0,
            sigma=bmb.Prior("Exponential", lam=1),
        )
    }
    
    any_debt_model = bmb.Model(formula=any_debt_model_formula, 
                            data=any_debt_model_df, 
                            family="bernoulli",
                            priors=any_debt_priors,
                            noncentered=True)
    
    
    pr_asset_model_cols = ['prank_assets',
                            'state',
                            'hh_income',
                            'age',
                            'race_eth',
                            'edu',
                            'tenure',
                            'household_type',
                            'male',
                            'metro',
                            'disability',
                            'class_worker',
                            'public_assistance',
                            'social_security',
                            'poverty',
                            'citizen',
                            'english_at_home',
                            'homevalue',
                            'race_eth_state',	
                            'race_eth_edu',	
                            'race_eth_age',	
                            'race_eth_income']

    pr_asset_model_df = sipp_model.select(pl.col(pr_asset_model_cols)).to_pandas().dropna(subset=pr_asset_model_cols)
    
    for col in ['state','hh_income','age','race_eth','edu','tenure',
            'household_type','class_worker','homevalue','race_eth_state',
            'race_eth_edu',	'race_eth_age',	'race_eth_income']:
        pr_asset_model_df[col] = pr_asset_model_df[col].astype('category')
        
    # change tenure from random effect to fixed effect
    pr_asset_model_formula = """
    prank_assets ~ male + metro + disability + class_worker + public_assistance
                + social_security + poverty + citizen + english_at_home + homevalue
                + tenure
                + (1|state) + (1|hh_income) + (1|age) + (1|race_eth) + (1|edu)
                + (1|household_type)
                + (1|race_eth:state) + (1|race_eth:edu) + (1|race_eth:age) + (1|race_eth:hh_income)
    """
    
    pr_asset_priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=2.5),
        "common": bmb.Prior("Normal", mu=0, sigma=1),
        "group_specific": bmb.Prior(
            "Normal",
            mu=0,
            sigma=bmb.Prior("Exponential", lam=1),
        )
    }
    
    pr_asset_model = bmb.Model(formula=pr_asset_model_formula, 
                            data=pr_asset_model_df, 
                            family="beta",
                            link='logit',
                            priors=pr_asset_priors,
                            noncentered=True)
    
    
    pr_debt_model_cols = ['prank_debts',
                        'state',
                        'hh_income',
                        'age',
                        'race_eth',
                        'edu',
                        'tenure',
                        'household_type',
                        'male',
                        'metro',
                        'disability',
                        'class_worker',
                        'public_assistance',
                        'social_security',
                        'poverty',
                        'citizen',
                        'english_at_home',
                        'homevalue',
                        'race_eth_state',	
                        'race_eth_edu',	
                        'race_eth_age',	
                        'race_eth_income']
    
    pr_debt_model_df = sipp_model.select(pl.col(pr_debt_model_cols)).to_pandas().dropna(subset=pr_debt_model_cols)
    
    eps = 1e-6
    pr_debt_model_df["prank_debts_adj"] = np.clip(pr_debt_model_df["prank_debts"], eps, 1 - eps)
    pr_debt_model_df.drop(['prank_debts'],axis=1,inplace=True)
    
    for col in ['state','hh_income','age','race_eth','edu','tenure',
            'household_type','class_worker','homevalue','race_eth_state',
            'race_eth_edu',	'race_eth_age',	'race_eth_income']:
        pr_debt_model_df[col] = pr_debt_model_df[col].astype('category')
        
    # change tenure from random effect to fixed effect
    pr_debt_model_formula = """
    prank_debts_adj ~ male + metro + disability + class_worker + public_assistance
                + social_security + poverty + citizen + english_at_home + homevalue
                + tenure
                + (1|state) + (1|hh_income) + (1|age) + (1|race_eth) + (1|edu)
                + (1|household_type)
                + (1|race_eth:state) + (1|race_eth:edu) + (1|race_eth:age) + (1|race_eth:hh_income)
    """
    
    pr_debt_priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=2.5),
        "common": bmb.Prior("Normal", mu=0, sigma=1),
        "group_specific": bmb.Prior(
            "Normal",
            mu=0,
            sigma=bmb.Prior("Exponential", lam=1),
        )
    }
    
    pr_debt_model = bmb.Model(formula=pr_debt_model_formula, 
                            data=pr_debt_model_df, 
                            family="beta",
                            link='logit',
                            priors=pr_debt_priors,
                            noncentered=True)
    
    return (any_asset_model,any_debt_model,pr_asset_model,pr_debt_model)

    
    