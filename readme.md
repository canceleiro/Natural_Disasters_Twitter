# Natural Disasters Twitter analysis
## by Javier Alonso Alonso

> In this project, it´s analysed disaster data from Figure Eight to build a model for an API that classifies disaster messages.

> The project contains a set of real messages that were sent during disaster events and it will be created a machine learning pipeline to categorize these events so that a message could be send to an appropriate disaster relief agency.

## Dataset

> There are two initial datasets:
> - Messages:it contains information on the twitter messages: the original twit, the translated into English twit and the source
> - Categories: it contains the categorization of all the messages into different categories


## ETL_Pipeline_Preparation

> The final dataframe is prepared from both initial dataframes and saved the result to a sqllite databese

> A py file was created process_data.py and for executing it I´ve ran 

> python process_data.py messages.csv categories.csv sqlite:///NaturalDisastersMsgs.db

## ML_Pipeline_Preparation

> The pipeline for building the machine learning algorithm is prepared in this section.

> A py file was created train_classifier.py and for executing it I´ve ran

> python train_classifier.py sqlite:///NaturalDisastersMsgs.db classifier.pkl

## Run.py

> For a web simulation it´s been created the Run.py

