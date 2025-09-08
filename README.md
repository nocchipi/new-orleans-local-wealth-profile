# new-orleans-local-wealth-profile

Data prep, modeling, and graphics for The Data Center's ["A Profile of Wealth in the New Orleans Metro" page](https://www.datacenterresearch.org/reports_analysis/a-profile-of-wealth-in-the-new-orleans-metro/) 

For a more explicit description of the method and choices, please see the accompanying technical paper linked to the wealth profile.


## Introduction & notes

Not all code in this repository is meant to be run locally. The code for the models is included, but will need a cloud compute resource to effectively run and store the model objects for use in later steps. There are many places in the code where I connect to a cloud data storage that someone outside of The Data Center will not be able to access, so you will have to run the code and write it to a place that you can access.

## Sequence of events

01_libraries_functions.R

02_data_cleaning.Rmd

03a_models.Rmd 

model_validation.Rmd

03b_poststrat.Rmd

04_graphics.Rmd


01_libraries_functions.R - calls all needed libraries and creates functions needed for the analysis. This is very likely to be updated with new functions associated to error calculations.

02_data_cleaning.Rmd - includes instructions for how to get the IPUMS and SIPP data, and indicator-by-indicator harmonizing documentation. 

03a_models.Rmd - This file was created for the purpose of code sharing, and has never been run as is. It used to exist as 4 separate notebooks that could run in the cloud environment concurrently. I do not suggest running the code in this document as 1 process, as it would take like 2 weeks.

model_validation.Rmd - This file doesn't have much documentation. It generates graphics and explanations for the technical paper for this project, and is part of our model validation. See the technical paper.

03b_poststrat.Rmd - After generating the model objects from the code in 03a_models, this file does the post-stratification of the models onto the local IPUMS data. This is where the final dataframe for the analysis & graphics gets created. This process is computationally intense for a short time. It runs on my PC, but nearly maxes out the CPU for a few minutes. There will be updates to this file in the future from The Data Center, including error calculations.

04_graphics.Rmd - This file generates the graphics for Metro New orleans included in the wealth profile.


