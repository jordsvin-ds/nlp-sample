# Code example: News Classificator
 
 
This repo contains a sample of a representative NLP-related Python code. 

This program is a Classificator of some already preparsed (by-word normalized, but not fully lemmatized) text files from miscellaneous news publications from open sources (the sample source files are in Russian, 200 items in each category). 

The zip-folder with `.txt` files (`News_Textfiles.zip`)  has to be unzipped and located in the project directory (same as the `demo_task.py` file) 

It subdivides all the news into 4 categories: `Culture`, `Economics`, `Sport` and `Miscellaneous` (the last is also based on the content of the training sample and not on the fact that a program is uncertain which group the selected text belongs to). The training sample is selected randomly.

The vectorisation of the words is done by the Tf-Idf Vectoriser, while the actual prediction is done with 4 different estimators: Logistic Regression, Random Forest, Support Vector Machines and XGBoost. 

The certain model is selectable via digit input. 

All 4 models later have their performance assessment, including averaged precision, recall and F1 measure; both for a model on the whole and by class. 

The  recommended Python interpreter is 3.7, however, the code contains nothing specific to a concrete interpreter version. 
