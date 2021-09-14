# System Design for a Data-Driven and Explainable Customer Sentiment Monitor Using IoT and Enterprise Data
This project was conducted at the [Machine Learning and Data Analytics Lab](https://www.mad.tf.fau.de/), Friedrich-Alexander-University Erlangen-Nuremberg (FAU) in cooperation with [Siemens Healthineers](https://www.siemens-healthineers.com/).

## Citation and Contact
You find the paper [here](https://ieeexplore.ieee.org/document/9520354).

If you use our work, please also cite the paper:
```
@ARTICLE{9520354,
  author={Nguyen, An and Foerstel, Stefan and Kittler, Thomas and Kurzyukov, Andrey and Schwinn, Leo and Zanca, Dario and Hipp, Tobias and Jun, Sun Da and Schrapp, Michael and Rothgang, Eva and Eskofier, Bjoern},
  journal={IEEE Access}, 
  title={System Design for a Data-Driven and Explainable Customer Sentiment Monitor Using IoT and Enterprise Data}, 
  year={2021},
  volume={9},
  number={},
  pages={117140-117152},
  doi={10.1109/ACCESS.2021.3106791}}
```
And for the dataset please cite:

```
@dataset{nguyen_an_2020_4383145,
  author       = {Nguyen, An and
                  Foerstel, Stefan and
                  Kittler, Thomas and
                  Kurzyukov, Andrey and
                  Schwinn, Leo and
                  Zanca, Dario and
                  Hipp, Tobias and
                  Da Jun, Sun and
                  Schrapp, Michael and
                  Rothgang, Eva and
                  Eskofier, Bjoern},
  title        = {{Industrial Benchmark Dataset for Customer 
                   Escalation Prediction}},
  month        = dec,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.4383145},
  url          = {https://doi.org/10.5281/zenodo.4383145}
}
```
The datasets are publicly available for research purposes. 

If you would like to get in touch, please contact an.nguyen@fau.de.
## Abstract

## Getting started
These instructions will get you a copy of the project up and running on your local machine
### Clone
Clone this repo to your local machine 
### Setup
Create environment using the requirements.txt
```
# using pip
pip install -r requirements.txt

# using Conda
conda create --name <env_name> --file requirements.txt
```
Activate environment
```
conda activate customer_sentiment_monitor 
```
Install src in the new environment 
```
pip install -e.
```

Register a notebook kernel
```
python -m ipykernel install --user --name=customer_sentiment_monitor
```

### Data
You can download the data via Zenodo [here](https://doi.org/10.5281/zenodo.4383145). Please put the files ```feature_matrix_LSTM.pickl``` and ```feature_matrix.pickle``` into the folder ```data/raw/``` in order to run the provided ```main.py```. 

### Run
**1.** To set the configurations, use the config.ini. The model_name, late_fusion_flag and feature_type from config.ini can be used to conduct the experiements. The rest of the configurations are exactly the same as in the paper.<br>
**2.** To run the weekly analysis, use the command below
```
python main.py
```
After each week the results are saved in data/interim/. If you want to continue training, use cont_week from config.ini.<br>
**3.** To visualize and evaluate data/results/results.pickle from weekly analysis, notebooks/Visualization.ipynb can be used.
The pdfs from visualization will be saved in data/visualizations/
## Contributors
[An Nguyen](https://www.mad.tf.fau.de/person/an-nguyen/), [Andrey Kurzyukov](https://github.com/SherlockKA), [Thomas Kittler](https://www.linkedin.com/in/dr-thomas-kittler-a379aa174/), [Stefan Foerstel](https://www.linkedin.com/in/stefan-foerstel/)
## Project Organization
------------
    ├── data
    │   ├── results                      <- The result from weekly analysis.
    │   ├── interim                      <- Intermediate data that has been transformed.
    │   ├── visualizations               <- The visualization of results from weekly analysis.
    │   └── raw                          <- The original, immutable data dump.
    ├── notebooks                        <- Jupyter notebooks. 
    ├── src                              <- Source code for use in this project.
        ├── config_parser.py             <- Script to parse configurations into dict.
        ├── general_helper_functions.py  <- Script with helper functions
        ├── train_Ensemble.py            <- Script to train XGBoost or RandomForest.
        ├── train_LSTM.py                <- Script to train LSTM.
        ├── visualize.py                 <- Script to visualize results from weekly analysis.
    ├── .gitignore                       <- Files that should be ignored.
    ├── README.md                        <- The top-level README for developers using this project.
    ├── config.ini                       <- Configurations.
    ├── requirement.txt                  <- The requirements file for reproducing the analysis environment, e.g.
    ├── main.py                          <- Main code.
    ├── setup.py                         <- makes project pip installable (pip install -e .) so src can be imported
--------


