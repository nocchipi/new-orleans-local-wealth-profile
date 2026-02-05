# Getting Started
* Install Miniconda
* Open a termial such as Windows Command Prompt
* Create a python environment: ```conda create --name no-lwp python=3.12```
* Activate environment: ```conda activate no-lwp```
* Navigate to the **python_data_cleaning** folder from the terminal and execute ```pip install -r requirements.txt```
* Create a .env file in the root directory of the code to store environmental variables used in the code.
  * Storing the API key used to get the IPUMS extract as **IPUMS_API_KEY**
* There is a boolean varaible in code called **DOWNLOAD_IPUMS** that if set to True will use the IPUMS API to downlaod the file.  If it is already downloaded set to False to avoid downloading it again.
* Download the SIPP PU and RW files and save file paths the environmental variables in the .env file called
  * **SIPP_PU_FILE_PATH**
  * **SIPP_RW_FILE_PATH**
* From R environments make sure **VIM**, **dplyr** and **srvyr** packages are installed. 
* Jupyter notebook called **data_cleaning.ipynb** has the data cleaning functionality in Python primarily using the Polars library.