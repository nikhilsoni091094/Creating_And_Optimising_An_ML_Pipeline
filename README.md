# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.

In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.

This model is then compared to an Azure AutoML run.

## Summary

The target variable here was to predict whether the customer will be interested in buying a product offered to them across marketing campaigns. The examples of independent variables are age, marital status, education, current loans and housing. 

- Approach:-

We have to use both Azure ML Studio HyperDrive and Azure AutoML to build upon a model and compare between the two approaches.

The best performing model was a VotingEnsemble with an Accuracy of 0.9171 , this one was calculated using the autoML Experiment.

## Scikit-learn Pipeline

- Pipeline details :-

Architecture: Virtual Machine General Purpose CPU Cluster Compute D-Series V2
Data: CSV Format, 21 columns, 32,950 data rows. The is loaded using a TabularDatasetFactory class, to acqurate the result the is cleaned using the function “clean_data” which is part of the script train.py
Classification algorithm: We use a Scikit-learn Logistic Regression Model with a parameter sampler
Hyperparameters: “C” which is the regularization parameter, “max-iter” which define the maximum number of iterations allowed

- Choosing a parameter sampler :-

We have used the RandomParameterSampling as parameter sampler.
ps = RandomParameterSampling( { '--C': uniform(0.05, 1), '--max_iter': choice(10, 30, 50, 70, 90) } )

- How to choose an early stopping policy :-

We have used BanditPolicy as an early stopping criteria.
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

## AutoML

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
