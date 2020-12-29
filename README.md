# Disaster Response Pipeline Project

## Installations

This project was created and run using Python version 3.0.
And it uses those libraries

*sys
*pandas
*numpy
*sqlalchemy
*pickle
*re
*nltk
*sklearn


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
   	 `python run.py`
    

3. Go to http://0.0.0.0:3001/


## Motivations
In this project we will analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

We will be creating a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

The project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 


## File Desriptions
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

    .Pynb : File contianing EDA and findings

    README.md
```
