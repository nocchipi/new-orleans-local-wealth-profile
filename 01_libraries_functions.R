  #libraries
  require("data.table")
  require("bit64")
  require("dplyr")
  require("srvyr")
  library(tidyverse)
  library(jamba)
  library(Hmisc)
  library(hutils)
  library(rstanarm)
  library(tidybayes)
  library(lme4)
  library(srvyr)
  library(bayesplot)
  library(htmlTable)
  library(here)
  library(rlang)
  library(purrr)
  library(furrr) #this is the same as purrr, but can go parallel.
  #also install future
  library(future)
  future::plan(multicore)
  library(ipumsr)
  library(sjPlot)
  library(htmlTable)
  library(ggrepel)
  library(broman)
  library(scales)
  library(spatstat)
  library(extrafont)
  library(VIM)
  #font_import()
  loadfonts(device="win")       #Register fonts for Windows bitmap output
  fonts()  
  source(here("inputs/datacenter_colors.R")) 
  


 predict_allmodels <- function(data, asset_class_model, asset_model, debt_class_model, debt_model, ndraw = 1){
   #first, predict having any assets 
   data_assets <- data %>%
      add_predicted_draws(asset_class_model, ndraws = ndraw) %>%
     rename(any_asset = .prediction) %>%
     ungroup() %>%
     select(-c(.row, .chain, .iteration, .draw))
      
   data_assets <- data_assets %>%
      add_predicted_draws(asset_model, ndraws = ndraw) %>%
      rename(asset_pred = .prediction,
             asset_draw = .draw) %>%
      ungroup() %>%
      mutate(asset_pred = case_when(any_asset == 0 ~ 0,
                                    any_asset == 1 ~ asset_pred)) %>%
      
      select(-c(.row, .chain, .iteration))
    
   #predict having debts
   data_debts <- data %>%
     add_predicted_draws(debt_class_model, ndraws = ndraw) %>%
     rename(any_debt = .prediction) %>%
     ungroup() %>%
     select(-c(.row, .chain, .iteration, .draw))
   
   
    data_debts <- data_debts %>% 
      add_predicted_draws(debt_model, ndraws = ndraw) %>%
      rename(debt_pred = .prediction,
             debt_draw = .draw) %>%
      ungroup() %>%
      mutate(debt_pred = case_when(any_debt == 0 ~ 0,
                                   any_debt == 1 ~ debt_pred)) %>%
      
      select(-c(.row, .chain, .iteration)) 
    
    data_assets$debt_draw <- data_debts$debt_draw
    data_assets$any_debt <- data_debts$any_debt
    data_assets$debt_pred <- data_debts$debt_pred
      
      
      
    return(data_assets)
 }
 


  quantile_plotdata <- function(survey_data, ps_data, group_variables){
  
    data_quantiles_longer <- survey_data %>% ungroup() %>%
      transmute(race_eth = race_eth,
                Q10 = var_quantile_q10,
                Q30 = var_quantile_q30,
                Q50 = var_quantile_q50,
                Q70 = var_quantile_q70,
                Q90 = var_quantile_q90,
  
                Q10_se = var_quantile_q10_se,
                Q30_se = var_quantile_q30_se,
                Q50_se = var_quantile_q50_se,
                Q70_se = var_quantile_q70_se,
                Q90_se = var_quantile_q90_se) %>%
      pivot_longer(cols = Q10:Q90_se, names_to = "var", values_to = "val")
  
    data_quant_vals <- data_quantiles_longer %>% filter(grepl("_se", var) == F)
    data_quant_se <- data_quantiles_longer %>% filter(grepl("_se", var)) %>% rename(se = val)
    data_quant_se$var <- str_replace_all(data_quant_se$var, "_se", "")
  
    data_quant <- data_quant_vals %>% left_join(data_quant_se, by = c("race_eth", "var"))
    return(data_quant)
  }
  
  ## Louisiana grouped by educational attainment
  
  distribution_quantile_chart <- function(geo1_data,
                                          geo2_data = NULL,
                                          grouping_variables = "race_eth") {
    
  }
  
  survey_dataprep <- function(survey_data, grouping_var, y_var) {

    SIPP_quantiles <- survey_data  %>% group_by(!!grouping_var) %>% srvyr::summarize(quantile = survey_quantile(!!y_var, c(.1,  .3, .5, .7, .9), na.rm = T))
    
    SIPP_quantiles_longer <- SIPP_quantiles %>% ungroup() %>%
      transmute(year = 2018,
                comp = "SIPP estimate",
                group = !!grouping_var,
                Q10 = quantile_q10,
                Q30 = quantile_q30,
                Q50 = quantile_q50,
                Q70 = quantile_q70,
                Q90 = quantile_q90,

                Q10_se = quantile_q10_se,
                Q30_se = quantile_q30_se,
                Q50_se = quantile_q50_se,
                Q70_se =quantile_q70_se,
                Q90_se = quantile_q90_se) %>%
      pivot_longer(cols = Q10:Q90_se, names_to = "var", values_to = "val")

    SIPP_quant_vals <- SIPP_quantiles_longer %>% filter(grepl("_se", var) == F)
    SIPP_quant_se <- SIPP_quantiles_longer %>% filter(grepl("_se", var)) %>%
      rename(se = val)
    SIPP_quant_se$var <- str_replace_all(SIPP_quant_se$var, "_se", "")

    SIPP_quant <- SIPP_quant_vals %>% left_join(SIPP_quant_se, by = c("year", "comp", "group", "var"))
    return(SIPP_quant)
  }
  
  ps_dataprep <- function(post_strat_data,
                                grouping_var, #must use quo()
                                y_var){ #must use quo()
    PS_quantiles <- post_strat_data %>% ungroup() %>%
      filter(state == 22) %>% group_by(!!grouping_var) %>% dplyr::summarize(quantile = quantileSE(!!y_var, c(.1,  .3, .5, .7, .9), na.rm = T))
    PS_quantiles <-cbind(group =  PS_quantiles %>% select(!!grouping_var) %>% as.data.frame(), PS_quantiles$quantile %>% as.data.frame())
    PS_quantiles$name <- rownames(PS_quantiles)
    rownames(PS_quantiles) <- NULL
    PS_quantiles <- PS_quantiles %>%
      rename(Q10 = V1, Q30 = V2, Q50 = V3, Q70 = V4, Q90 = V5) %>%
      mutate(name = case_when(grepl("quantile", name) ~ "quantile", grepl("SE", name) ~ "se"))
    
   
    PS_quantiles_longer <- PS_quantiles %>%
      ungroup() %>%
      pivot_longer(cols = -c(!!grouping_var, name), names_to = "var", values_to = "val") %>%
      pivot_wider(names_from = name, values_from = val) %>%
      transmute(year = 2018,
                comp = "MRP estimate",
                group = !!grouping_var,
                var = var,
                val = as.numeric(quantile),
                se = as.numeric(se))


    return(PS_quantiles_longer)
  }
  
quantile_plot <- function(panel1_data, panel2_data = NULL, filter_vec = NULL, title, subtitle, xlab, ylab, colorlab){
  quant_data <- panel1_data
  if(is.null(panel2_data) == F){
    quant_data <- rbind(panel1_data, panel2_data)
  }
  
  if(is.null(filter_vec) == F){
    quant_data <- quant_data %>% filter(group %in% as.vector(filter_vec))
  }
  
  p <- ggplot(quant_data, aes(x=var, y=val, group= group, color= group)) + 
    geom_line() +
    geom_point()+
    # geom_errorbar(aes(ymin=val-(2*se), ymax=val+(2*se)), width=.3,
    #               position=position_dodge(0.05)) + 
    themeDC_horizontal() +
    scale_color_manual(values = race_DCcolors) +
    labs(title = title,
         subtitle = subtitle,
         x = xlab,
         y = ylab,
         color = colorlab) + 
    facet_wrap(~comp)
  return(p)
}
  

  
  plus <- function(x) {
    if(all(is.na(x))){
      c(x[0],NA)} else {
        sum(x,na.rm = TRUE)}
  }
  