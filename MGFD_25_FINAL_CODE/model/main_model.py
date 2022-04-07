import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
#Model import
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

class data_cleaning():
    def drop_na(self):
        return self.dropna(axis=0)
    
    def remove_outliers(self, df, cri):
        """df is the data frame that you want to clean

        cri is the judging critira of which you want to remove
        """
        return None
        
class figure_drawing():
    
    """This class method is to draw datasets
    
    """
    def scatter_plot(self, data, dimensions, color):
        """Scatterplot matrix
            - Ignore self for now 
            - Data should be the data set in pd dataframe
            - Dimensions are the variables wanted to be drawn
            - Color is the color code judging critria
        """
        fig = px.scatter_matrix(data, dimensions=dimensions, \
            labels={col:col.replace('_', ' ') for col in data.columns}, \
                height=900, color=color, \
                    color_continuous_scale=px.colors.diverging.Tealrose)
        fig.show
        
class models():
    """Models class
    """
    def model_run(self, df, model, name):
        """
        """
        x_train, x_test, y_train, y_test = train_test_split(df.drop('loan_status',axis=1),df['loan_status'], random_state=0, test_size=.20)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        preds_proba = model.predict_proba(x_test)
        print('                   ', name, '\n', \
            classification_report(y_test, model.predict(x_test)))
#class models():
if __name__ == "__main__":
    #0 (no default) or 1 (default)
    df = pd.read_csv('/Users/robkang/Documents/MGFD_25_FINAL_CODE/model/credit_risk_dataset.csv')
    print(df)
    figure_drawing.scatter_plot("", df, ["person_age","person_income","person_emp_length", \
                           "loan_amnt","loan_int_rate"], "loan_status")
    #Drop nas
    df = df.dropna()
    fig = px.parallel_categories(df, color_continuous_scale=px.colors.sequential.RdBu, color="loan_status",\
        dimensions=['person_home_ownership','loan_intent', "loan_grade", 'cb_person_default_on_file'], labels={col:col.replace('_', ' ') for col in df.columns})
    fig.show()
    #Encoding dummies:
    df = pd.get_dummies(data=df,columns=['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file'])
    #KNN
    knn = KNeighborsClassifier(n_neighbors=151)
    models.model_run("", df, knn, name='KNN')
    #Logistic Regression
    lg = LogisticRegression(random_state=0)
    models.model_run("", df,lg, 'Logistic Regression')
    

    

    x_train, x_test, y_train, y_test = train_test_split(df.drop('loan_status',axis=1),df['loan_status'], random_state=70, test_size=.30)

    #ROC AUC
    fig = plt.figure(figsize=(14,10))
    plt.plot([0, 1], [0, 1],'r--')
    #KNN
    preds_proba_knn = knn.predict_proba(x_test)
    probsknn = preds_proba_knn[:, 1]
    fpr, tpr, thresh = roc_curve(y_test, probsknn)
    aucknn = roc_auc_score(y_test, probsknn)
    plt.plot(fpr, tpr, label=f'KNN, AUC = {str(round(aucknn,3))}')
    #Logistic Regression
    preds_proba_lg = lg.predict_proba(x_test)
    probslg = preds_proba_lg[:, 1]
    fpr, tpr, thresh = roc_curve(y_test, probslg)
    auclg = roc_auc_score(y_test, probslg)
    plt.plot(fpr, tpr, label=f'Logistic Regression, AUC = {str(round(auclg,3))}')
    
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title("ROC curve")
    plt.rcParams['axes.titlesize'] = 18
    plt.legend()
    plt.show()