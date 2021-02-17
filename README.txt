# Ingredient Tagger

## Project Definition

Ingredient Tagger is a capstone project for Udacity Machine Learning Engineer Nanodegree. It is a service for tagging ingredient name, quantity and units in ingredient phrased commonly seen in the cooking recipes. Full description of the project can be found [here]('Capstone Project Report.pdf')


## Application
Deployed application is hosted at https://wsexg6g746.execute-api.us-west-2.amazonaws.com/prod
It accepts the list of ingredient phrases in json format and returnes the list of model prediction results. 

### Example:
```bash
curl -X POST --data '["1 large onion, sliced", "1 1/2 tablespoons finely minced garlic", "2 peeled and cubed potatoes"]'  https://wsexg6g746.execute-api.us-west-2.amazonaws.com/prod
```

## Setup
The project is developed for AWS, and can be deployed using AWS SageMaker, Lambda and API Gateway. To deploy the project on AWS:
 - create a SageMaker notebook with this repository attached
 - run Prepare_Data.ipynb notebook to generate data for training
 - run Train_the_Model.ipynb to train the model and deploy it on an endpoint
 - creat IAM role for the Lambda function with SageMaker access
 - set up a Lambda function with the code from the lambda_function.py file as a handler. Make sure it is edited, so that it calls the newly created endpoint
 - set up an AWS API Gateway for a public access to the service
 - the notebook for different model evaluation (Model_Selection.ipynb) can be run on Gooogle Colab or a machine with GPU.

### Dependencies:
 Models are developed in Pytorch, other dependencies are listed in the [requrements.txt](source/requirements.txt) file


