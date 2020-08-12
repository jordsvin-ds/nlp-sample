
from sklearn.model_selection import  cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from time import  sleep


# a separate class for input
class Modelling_Procedure:
    # a dictionary containing the names of methods and fitted-objects
    reg_vocabulary = {0: ('log_reg', 'Logistic Regression', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=120)),
                      1: ('random_forest', 'Random Forest', RandomForestClassifier()),
                      2: ('xgboost', 'XGBClassifier' ,xgb.XGBClassifier(objective='multi:softmax')),
                      3: ('svm', 'SVM', SVC(kernel='rbf', random_state=1))}
    dig_vocabulary = {0: ('tfidf', 'Tf-idf', TfidfVectorizer()) }

    def __init__(self, regr_code, dig_method_code):
        self.regression_full_name = self.reg_vocabulary[regr_code][1]
        self.regression_short_name = self.reg_vocabulary[regr_code][0]
        self.regression_model_object = self.reg_vocabulary[regr_code][2]
        self.digfullname = self.dig_vocabulary[dig_method_code][1]
        self.digshortname = self.dig_vocabulary[dig_method_code][0]
        self.digmodelobject = self.dig_vocabulary[dig_method_code][2]




def validate(X, y):

    prec_matrix = []
    rec_matrix = []
    obtain_results(X, y, rec_matrix, prec_matrix)
    scoring = cross_validate(regression_model, X, y, cv = 6, scoring = ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro'))
    return  scoring


def output(scorings):
    print(
        'Average Precision, Recall and F-Measure for ' + full_procedure.digfullname + ' & '
        + full_procedure.regression_full_name + ' method respectively: ')

    new_scoring = [scorings['test_precision_macro'],
                   scorings['test_recall_macro'],
                   scorings['test_f1_macro']]

    for item in new_scoring:
        print(round(sum(item) / len(item), 4))
    print('\n')


    
# a function importing tematic texts from respective folders located within the main folder.

def make_X_and_y_matrix(folders):
    paths_to_class = {"culture\\ind": 'Culture',
                         "econ\\ind": 'Economy',
                         "sport\\ind": 'Sport',
                         "other\\ind": 'Miscellaneous'}
    # y are text classes, X are texts
    y, x = [], []
    for i, address in enumerate(folders):
        for ind in range(0, 200):

            y.append(paths_to_class[address])
            filename = address + str(ind) + '_proc.txt'
            f = open(filename, 'r', encoding='UTF-8')
            x.append(f.read())
            f.close()
    return x, y



def obtain_results(X, y, rec_matrix, prec_matrix):
    y_real = np.asarray(y)
    y_pred = cross_val_predict(regression_model, X, y, cv=6)
    inaccuracy_matrix = confusion_matrix(y_real, y_pred)
    recall_list = []

    print('The recall statistics for Culture, Economics, Sport and Miscellaneous respectively are as follows: ')

    for i in range(4):
        recall_list.append(round(inaccuracy_matrix[i][i] / sum(inaccuracy_matrix[i]), 3))
    print(recall_list)
    rec_matrix.append(recall_list)
    precision_list = []
    trans_inaccuracy_matrix = inaccuracy_matrix.transpose()

    print('The precision statistics for Culture, Economics, Sport and Miscellaneous respectively are as follows: ')

    for i in range(4):
        precision_list.append(round(trans_inaccuracy_matrix[i][i] / sum(trans_inaccuracy_matrix[i]), 3))
    print(precision_list)
    prec_matrix.append(precision_list)






if __name__ =="__main__":

    ## --MAIN CODE--

    files = ["culture\\ind", "econ\\ind", "sport\\ind", "other\\ind"]

    # get data
    X, y = make_X_and_y_matrix(files)

    sleep(4)
    print('\n')
    #a list of methods (extendable) input
    regressions_message = 'Type a digit correspondent to a quantitative method you would like to apply: \n0. Logistic Regression \n1. Random Forest \n2. XGBoost \n3. SVM \n'

    full_procedure = Modelling_Procedure(int(input(regressions_message)), 0)

    print('(wait, especially for SVM and XGB)')
    regression_model = Pipeline(steps=[(full_procedure.digshortname, full_procedure.digmodelobject),
                                       (full_procedure.regression_short_name, full_procedure.regression_model_object)])
    scorings = validate(X, y)

    output(scorings=scorings)






