# PolAna
<div id="top"></div>
<br />
<div align="center">
  <h3 align="center">DSR Portfolio Project Political Analyser</h3>
  </p><p align="center">
  
</div>

<div align="center">
<img src="https://thumbs.dreamstime.com/z/fake-news-concept-user-click-text-analyze-which-real-109649276.jpg" alt="Logo" width="700" height="500">
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The Data Science Retreat intensive programming boot camp culminates in a portfolio project. The project we have decided to pursue is to address the growing manipulation and influencing of voters through the use of online media.

Our main question was:

Is there a way for a user to monitor whether they are being influenced by media articles?

Media bias has contributed to people becoming more politically polarized. When we can’t identify, understand and appreciate diverse perspectives, we are more likely to be manipulated into thinking or voting a certain way.

The online domain is a slippery slope where a person could venture far away from their original political disposition within a matter of clicks i.e. going "deeper down the rabbit hole".

This project aims to create a personal media tracker, based on a person's online reading history, to notify the user that they are viewing unusual material and run the risk of being influenced or manipulated.

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With
...love... no, not really.

The project was implemented using the following packages:

* Numpy
* Pandas
* MatplotLib
* CategoryEncoders (for binary encoders)
* Feature_Engine_Encoding (for frequency encoders)
* OS
* SKLearn
* Lightgbm
* Catboost
* XGBoost
* Pickle
* Time
* Datetime
* Glob
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

In order to obtain the required data, you will need to create an account on Driven Data (https://www.drivendata.org/). 
After having created an account, the data can be downloaded from the following page:

https://www.drivendata.org/competitions/57/nepal-earthquake/

### Prerequisites

Ensure that the required packages listed under "Built with" have been installed and are up to date.


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show how winning is done.

<div align="center">
<img src="https://www.memesmonkey.com/images/memesmonkey/f3/f317606f2702e64e2284cc633e849e4b.jpeg" alt="Logo" width="700" height="500">
</div>

The training data labels and values have to be imported and merged together into one dataset. This data was then used for data visualisation:
<div align="center">
    <img src="src/visualization/Subplots 1.png" alt="Logo" width="500" height="500">
    <img src="src/visualization/Subplots 2.png" width="500" height="500">
    <img src="src/visualization/Subplots 3.png" alt="Logo" width="500" height="500">
    <img src="src/visualization/Subplots 4.png" alt="Logo" width="500" height="500">
    <img src="src/visualization/Subplots 5.png" alt="Logo" width="500" height="500">
</div>

Subsequently, the features for the model need to be built and selected. The code specifies two iterations, one vanilla with no changes to the code and having dropped all categorical and low correlation data, as well as a routine build where categorical data were encoded using a binary and frequency encoder, as well as some modification to the data, such as normalisation and removal of outliers.

To use this routine on a different data set, You will have to edit build_features and value_column_string in split_train_dataset as this is specific to Richter's Predictor: Modeling Earthquake Damage
https://www.drivendata.org/competitions/57/nepal-earthquake

use_vanilla_data is made as a global variable as it needs to be consistent for train and test datasets

<p align="right">(<a href="#top">back to top</a>)</p>

Split the train dataframe according to value_column_string and train_test_split params.

The routine will go through all sklearn classifiers from:
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

and additional:
sklearn.tree.DecisionTree
sklearn.linear_models.SGDClassifier
XGBoost
CatBoost
LGBM
 
Use only the baseline model set with use_baseline_models=True (default:False)

Mutually exclusive options for hyperparameter optimization:
Enable GridSearchCV for all models with grid_cv=True (default: False)
Enable RandomSearchCV for all models with random_cv=True (default: False)

GridSearchCV parameter rages done with rule of thumb adequately to a Classifier class
Starting RandomSearchCV were done with a rule of thumb adequately to a Classifier class
Hyperparameter optimization takes a considerable amount of time so use with caution 

The method will test and score the model with F1 micro and macro averaged score
Additionally a cross validation score will be generated for the train dataset

Create the test dataset to generate results for upload

Apply test dataset to all trained models ang generate results.
Results in separate files per model found in ../data/results

Execute, run away, and pray for the best and pray to God you haven't ruined Shishtoff's code.

<!-- ROADMAP -->
## Roadmap

- [x] Download data
- [x] Review data for patterns and/or discrepancies
- [x] Clean data
- [x] Build and select features for use in model
- [x] Select model and fit on training data
- [x] Use the model to make predictions for test data
- [ ] Call it a day and go grab a beer

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

A special shout out to my boy, Shishtoff and Paul and last, but not least the sweet flower of the office, me, David.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Used for learning purposes only. Not to be distributed.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Special shout out to the DSR! Go Quesadas!

<p align="right">(<a href="#top">back to top</a>)</p>