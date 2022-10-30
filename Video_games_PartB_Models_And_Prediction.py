import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, davies_bouldin_score, confusion_matrix, silhouette_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
import VisualizeNN as VisNN
from sklearn.svm import SVC
from sklearn.metrics import precision_score , roc_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, davies_bouldin_score , classification_report
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, cross_val_score
from tabulate import tabulate
from pandas.plotting import scatter_matrix
from numpy import mean, std
from scipy.stats import norm
from sklearn.svm import SVR


##########################################################################
# ***********----------- Model Training ------------******************
def transform_to_bin():
    est = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
    df['EU_Sales'] = est.fit_transform(df[['EU_Sales']])


# -----Read the data------
df = pd.read_csv('XY_train_impr.csv', header=0)
pd.options.display.max_columns = None
transform_to_bin()
# df = df.astype(
#     dtype={'Year_of_Release': 'category', 'Genre': 'category', 'Rating': 'category', 'Platform_General': 'category',
#            'General_Sales': 'float', 'Critic_Weight': 'float', 'User_Weight': 'float', 'EU_Sales': 'category'})

# ---- Split our Data to X_train and Y_train from the united csv ----
temp_df = df.copy()
Y_train = temp_df['EU_Sales']
temp_df.drop('EU_Sales', axis='columns', inplace=True)
X_train = temp_df
# ----K-fold------
KF = KFold(n_splits=10)

# ***********----------- Decision Trees ------------******************
# ----Dummy variables for decision trees---
X_train = pd.get_dummies(X_train)

# ----Defualt Decision Tree Without Tuning----
model_dt = DecisionTreeClassifier(random_state=42)
train_result_dt = pd.DataFrame()
valid_result_dt = pd.DataFrame()
for train_index , val_index in KF.split(X_train):
    train_result_md = pd.DataFrame(None)
    valid_result_md = pd.DataFrame(None)
    model_dt.fit(X_train.iloc[train_index] , Y_train.iloc[train_index])
    acc_train = accuracy_score(Y_train.iloc[train_index] ,  model_dt.predict(X_train.iloc[train_index])) #accuracy of the train
    acc_val = accuracy_score(Y_train[val_index],  model_dt.predict(X_train.iloc[val_index])) #accuracy of the validation
    train_result_dt = train_result_dt.append({'Train accuracy':acc_train},ignore_index=True)
    valid_result_dt = valid_result_dt.append({'Validation accuracy':acc_val},ignore_index=True)
avg_acc_train = train_result_dt.mean()
avg_acc_val = valid_result_dt.mean()
print(avg_acc_train)
print(avg_acc_val)

#----Hyperparameters Tuning----
param_grid = {'max_depth': np.arange(1, 50, 1),
              'criterion': ['entropy', 'gini'],
              'ccp_alpha': np.arange(0, 1, 0.05)
             }
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, refit=True, cv=KF , return_train_score=True )
grid_search.fit(X_train,Y_train)
Results = pd.DataFrame(grid_search.cv_results_)
print('The best parameters are:' ,grid_search.best_params_)
y_val = Results['mean_test_score']
y_train = Results['mean_train_score']
results_grid_search1 = pd.DataFrame(Results).sort_values('rank_test_score')[['params', 'mean_test_score']]
results_grid_search2 = pd.DataFrame(Results).sort_values('mean_train_score',ascending=False)[['params', 'mean_train_score']]
headers_val = ["Number", "Parameters", "Validation score"]
headers_train = ["Number", "Parameters", "Train score"]
print(tabulate(results_grid_search1, headers = headers_val, tablefmt = "grid"))
print(tabulate(results_grid_search2, headers = headers_train, tablefmt = "grid"))
#plot accuracy for validation
plt.plot(y_val)
plt.ylabel('Validation accuracy', fontsize = 10)
plt.xlabel('Iteration', fontsize = 10)
plt.title('Accuracy for each Iteration', fontsize = 20)
plt.show()
#plot accuracy for train
plt.plot(y_train)
plt.ylabel('Train accuracy', fontsize = 10)
plt.xlabel('Iteration', fontsize = 10)
plt.title('Accuracy for each Iteration', fontsize = 20)
plt.show()

#----Max depth influence on accuracy----
max_depth_range= np.arange(1, 50 , 1)
result_md = pd.DataFrame()
train_result_md = pd.DataFrame()
valid_result_md = pd.DataFrame()
for max_depth in max_depth_range:
    train_result_md = pd.DataFrame(None)
    valid_result_md = pd.DataFrame(None)
    for train_index, val_index in KF.split(X_train):
        model_md=DecisionTreeClassifier(criterion='gini', max_depth=max_depth, random_state=42) #Our best parameters after tuning!
        model_md.fit(X_train.iloc[train_index], Y_train.iloc[train_index])
        acc_train_md = accuracy_score(Y_train.iloc[train_index], model_md.predict(X_train.iloc[train_index]))  # accuracy of the train
        acc_val_md = accuracy_score(Y_train[val_index], model_md.predict(X_train.iloc[val_index]))  # accuracy of the validation
        train_result_md = train_result_md.append({'Train accuracy': acc_train_md}, ignore_index=True)
        valid_result_md = valid_result_md.append({'Validation accuracy': acc_val_md}, ignore_index=True)
    avg_train_md = train_result_md.mean()
    avg_val_md = valid_result_md.mean()
    result_md = result_md.append({'max_depth': max_depth,
                                'train_acc':avg_train_md,
                                 'val_acc':avg_val_md}, ignore_index=True)

plt.figure(figsize=(13, 4))
plt.plot(result_md['max_depth'], result_md['train_acc'], marker='o', markersize=4)
plt.plot(result_md['max_depth'], result_md['val_acc'], marker='o', markersize=4)
plt.title("Infulence of Max depth on accuracy")
plt.ylabel('Accuracy', fontsize = 10)
plt.xlabel('Max depth', fontsize = 10)
plt.legend(['Train accuracy', 'Validation accuracy'])
plt.show()

#----Ccp alpah influence on accuracy----
ccp_alpha_range= np.arange(0, 1, 0.05)
result_ccp = pd.DataFrame()
train_result_ccp = pd.DataFrame()
valid_result_ccp = pd.DataFrame()
for ccp_alpha in ccp_alpha_range:
    train_result_ccp = pd.DataFrame(None)
    valid_result_ccp = pd.DataFrame(None)
    for train_index, val_index in KF.split(X_train):
        model_ccp = DecisionTreeClassifier(criterion='gini', max_depth=6, ccp_alpha=ccp_alpha,random_state=42)
        model_ccp.fit(X_train.iloc[train_index], Y_train.iloc[train_index])
        acc_train_ccp = accuracy_score(Y_train.iloc[train_index], model_ccp.predict(X_train.iloc[train_index]))  # accuracy of the train
        acc_val_ccp = accuracy_score(Y_train[val_index], model_ccp.predict(X_train.iloc[val_index]))  # accuracy of the validation
        train_result_ccp = train_result_ccp.append({'Train accuracy': acc_train_ccp}, ignore_index=True)
        valid_result_ccp = valid_result_ccp.append({'Validation accuracy': acc_val_ccp}, ignore_index=True)
    avg_train_ccp = train_result_ccp.mean()
    avg_val_ccp = valid_result_ccp.mean()
    result_ccp = result_ccp.append({'max_depth': ccp_alpha,
                                'train_acc':avg_train_ccp,
                                 'val_acc':avg_val_ccp}, ignore_index=True)

plt.figure(figsize=(13, 4))
plt.plot(result_ccp['max_depth'], result_ccp['train_acc'], marker='o', markersize=4)
plt.plot(result_ccp['max_depth'], result_ccp['val_acc'], marker='o', markersize=4)
plt.title("Infulence of Ccp alpha on accuracy")
plt.ylabel('Accuracy', fontsize = 10)
plt.xlabel('Ccp alpha', fontsize = 10)
plt.legend(['Train accuracy', 'Validation accuracy'])
plt.show()

#----Criterion influence on accuracy----
criterion_vec= ['entropy', 'gini']
result_cr = pd.DataFrame()
train_result_cr = pd.DataFrame()
valid_result_cr = pd.DataFrame()
for criterion in criterion_vec:
    train_result_cr = pd.DataFrame(None)
    valid_result_cr = pd.DataFrame(None)
    for train_index, val_index in KF.split(X_train):
        model_cr = DecisionTreeClassifier(criterion=criterion, max_depth=6, ccp_alpha=0,random_state=42)
        model_cr.fit(X_train.iloc[train_index], Y_train.iloc[train_index])
        acc_train_cr = accuracy_score(Y_train.iloc[train_index], model_cr.predict(X_train.iloc[train_index]))  # accuracy of the train
        acc_val_cr = accuracy_score(Y_train[val_index], model_cr.predict(X_train.iloc[val_index]))  # accuracy of the validation
        train_result_cr = train_result_cr.append({'Train accuracy': acc_train_cr}, ignore_index=True)
        valid_result_cr = valid_result_cr.append({'Validation accuracy': acc_val_cr}, ignore_index=True)
    avg_train_cr = train_result_cr.mean()
    avg_val_cr = valid_result_cr.mean()
    result_cr = result_cr.append({'max_depth': criterion,
                                'train_acc':avg_train_cr,
                                 'val_acc':avg_val_cr}, ignore_index=True)

plt.figure(figsize=(13, 4))
plt.plot(result_cr['max_depth'], result_cr['train_acc'], marker='o', markersize=4)0.7761194029850746
plt.plot(result_cr['max_depth'], result_cr['val_acc'], marker='o', markersize=4)
plt.title("Infulence of Criterion on accuracy")
plt.ylabel('Accuracy', fontsize = 10)
plt.xlabel('Criterion', fontsize = 10)
plt.legend(['Train accuracy', 'Validation accuracy'])
plt.show()

#----Best Decision Tree After Tuning----
model_bt = DecisionTreeClassifier(criterion='gini', max_depth=9, random_state=42)
train_result_bt = pd.DataFrame()
valid_result_bt = pd.DataFrame()
for train_index , val_index in KF.split(X_train):
    model_bt.fit(X_train.iloc[train_index] , Y_train.iloc[train_index])
    acc_train = accuracy_score(Y_train.iloc[train_index] ,  model_bt.predict(X_train.iloc[train_index])) #accuracy of the train
    acc_val = accuracy_score(Y_train.iloc[val_index],  model_bt.predict(X_train.iloc[val_index])) #accuracy of the validation
    train_result_bt = train_result_bt.append({'Train Best accuracy':acc_train},ignore_index=True)
    valid_result_bt = valid_result_bt.append({'Validation Best accuracy':acc_val},ignore_index=True)
    # plt.figure(figsize=(11,6))
    # plot_tree(model_bt, filled=True, class_names=['0', '1'], feature_names=X_train.columns, fontsize=5 , max_depth=3)
    # plt.show()
avg_acc_train_bt = train_result_bt.mean()
avg_acc_val_bt = valid_result_bt.mean()
print(avg_acc_train_bt)
print(avg_acc_val_bt)

#-----Feature importances------
feature_importance = model_bt.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0])
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx], fontsize=10)
plt.title('Feature Importance', fontsize = 20)
plt.show()


# ***********----------- ANN ------------******************
scaler = MinMaxScaler()
std_scaler = StandardScaler()
loss_curve_flag = True

X_train_s = X_train.copy()
std_scaler.fit(X_train.copy())
X_train_s['General_Sales'] = scaler.fit_transform(X_train[['General_Sales']])

ann_model = MLPClassifier(random_state=42)

ann_train_result = pd.DataFrame()
ann_valid_result = pd.DataFrame()
for train_idx, val_idx in KF.split(X_train_s):
    x_train = X_train_s.iloc[train_idx]
    y_train = Y_train.iloc[train_idx]
    x_test = X_train_s.iloc[val_idx]
    y_test = Y_train.iloc[val_idx]

    ann_model.fit(x_train, y_train)
    acc_train = accuracy_score(y_train, ann_model.predict(x_train))
    acc_val = accuracy_score(y_test, ann_model.predict(x_test))

    # xtest = np.linspace(-5, 5, 50)
    # ytest = np.linspace(-5, 5, 50)
    # predictions = pd.DataFrame()
    # for x in tqdm(xtest):
    #     for y in ytest:
    #         pred = ann_model.predict(std_scaler.transform(np.array([x, y]).reshape(-1, 2)))[0]
    #         predictions = predictions.append(dict(X1=x, X2=y, y=pred), ignore_index=True)
    #
    # print(f"Train Accuracy: {acc_train:.3f}")
    # print(f"Validation Accuracy: {acc_val:.3f}")
    print(confusion_matrix(y_true=y_train, y_pred=ann_model.predict(x_train)))
    #
    # train_data = X_train_s.append(Y_train)
    # plt.scatter(x=predictions[predictions.y == 0]['X1'], y=predictions[predictions.y == 0]['X2'], c='powderblue')
    # plt.scatter(x=predictions[predictions.y == 1]['X1'], y=predictions[predictions.y == 1]['X2'], c='ivory')
    # sns.scatterplot(x='X1', y='X2', data=train_data, hue='y')
    # plt.show()
    if loss_curve_flag:
        plt.plot(ann_model.loss_curve_)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()

        network_structure = np.hstack(([x_train.shape[1]], np.asarray(ann_model.hidden_layer_sizes), 2))
        network = VisNN.DrawNN(network_structure)
        network.draw()


        loss_curve_flag = False

    ann_train_result = ann_train_result.append({'train_acc': acc_train}, ignore_index=True)
    ann_valid_result = ann_valid_result.append({'val_acc': acc_val}, ignore_index=True)

print(ann_train_result)
print(ann_valid_result)
avg_acc_train = ann_train_result.mean()
avg_acc_val = ann_valid_result.mean()
print(avg_acc_train)
print(avg_acc_val)

# ###_____________** ANN Hyper-parameters tuning **______________##
param_grid = {'hidden_layer_sizes': [(7,), (7, 7), (8,), (8, 8), (9,), (9, 9), (10,), (10, 10),(11,), (11,11), (12,), (12, 12),  (14,), (14, 14),
                                     (7, 9), (9, 7), (8, 10), (10, 8), (10, 12), (12, 10), (9, 11), (11, 9)],
              'max_iter': [500],
              'solver': ['lbfgs']
              }

grid_search = GridSearchCV(estimator=MLPClassifier(random_state=42, verbose=True, max_iter=500), param_grid=param_grid,
                           refit=True, cv=KF, return_train_score=True)
grid_search.fit(X_train_s, Y_train)
Results = pd.DataFrame(grid_search.cv_results_)
print('The best parameters are:', grid_search.best_params_)
y_val = Results['mean_test_score']
y_train = Results['mean_train_score']
results_grid_search1 = pd.DataFrame(Results).sort_values('rank_test_score')[['params', 'mean_test_score']]
results_grid_search2 = pd.DataFrame(Results).sort_values('mean_train_score', ascending=False)[
    ['params', 'mean_train_score']]
headers_val = ["Number", "Parameters", "Validation score"]
headers_train = ["Number", "Parameters", "Train score"]
print(tabulate(results_grid_search1, headers=headers_val, tablefmt="grid"))
print(tabulate(results_grid_search2, headers=headers_train, tablefmt="grid"))
plt.plot(y_val)
plt.ylabel('Validation accuracy', fontsize=10)
plt.xlabel('Iteration', fontsize=10)
plt.title('Accuracy for each Iteration', fontsize=20)
plt.show()



# #####_____________Visualization of tuning_____________########
----Number of neurons in hidden layer influence on accuracy----
param_vec = ['lbfgs', 'adam', 'sgd']
result = pd.DataFrame()
train_result = pd.DataFrame()
valid_result = pd.DataFrame()
for element in param_vec:
    train_result = pd.DataFrame(None)
    valid_result = pd.DataFrame(None)
    model_ = MLPClassifier(random_state=42, hidden_layer_sizes=(9, 9), max_iter=500, solver=element)
    for train_index, val_index in KF.split(X_train):
        x_train = X_train_s.iloc[train_index]
        y_train = Y_train.iloc[train_index]
        x_test = X_train_s.iloc[val_index]
        y_test = Y_train.iloc[val_index]

        model_.fit(x_train, y_train)
        acc_train = accuracy_score(y_train, model_.predict(x_train))
        acc_val = accuracy_score(y_test, model_.predict(x_test))
        train_result = train_result.append({'Train accuracy': acc_train}, ignore_index=True)
        valid_result = valid_result.append({'Validation accuracy': acc_val}, ignore_index=True)
    avg_train = train_result.mean()
    avg_val = valid_result.mean()
    result = result.append({'max_iter': element,
                            'train_acc': avg_train,
                            'val_acc': avg_val}, ignore_index=True)

plt.figure(figsize=(13, 4))
plt.plot(param_vec, result['train_acc'], marker='o', markersize=4)
plt.plot(param_vec, result['val_acc'], marker='o', markersize=4)
plt.title("Influence of solver on accuracy")
plt.ylabel('Accuracy', fontsize=10)
plt.xlabel('solver', fontsize=10)
plt.legend(['Train accuracy', 'Validation accuracy'])
plt.show()


# ######___________ ANN model with best config____________##############


ann_model = MLPClassifier(random_state=42, hidden_layer_sizes=10, max_iter=500, solver='lbfgs')

ann_train_result = pd.DataFrame()
ann_valid_result = pd.DataFrame()
for train_idx, val_idx in KF.split(X_train_s):
    x_train = X_train_s.iloc[train_idx]
    y_train = Y_train.iloc[train_idx]
    x_test = X_train_s.iloc[val_idx]
    y_test = Y_train.iloc[val_idx]

    ann_model.fit(x_train, y_train)
    acc_train = accuracy_score(y_train, ann_model.predict(x_train))
    acc_val = accuracy_score(y_test, ann_model.predict(x_test))
    if loss_curve_flag:
        # plt.plot(ann_model.l)
        # plt.xlabel("Iteration")
        # plt.ylabel("Loss")
        # plt.show()
        #
        # network_structure = np.hstack(([x_train.shape[1]], np.asarray(ann_model.hidden_layer_sizes), 2))
        # network = VisNN.DrawNN(network_structure)
        # network.draw()

        loss_curve_flag = False

    ann_train_result = ann_train_result.append({'train_acc': acc_train}, ignore_index=True)
    ann_valid_result = ann_valid_result.append({'val_acc': acc_val}, ignore_index=True)

print(ann_train_result)
print(ann_valid_result)
avg_acc_train = ann_train_result.mean()
avg_acc_val = ann_valid_result.mean()
print(avg_acc_train)
print(avg_acc_val)

#   ### Find largest unsureness
#
ann_model.fit(X_train_s, Y_train)
print(accuracy_score(Y_train, ann_model.predict(X_train_s)))
pp = ann_model.predict_proba(X_train_s)
print(pp)
pp_0 = pp[:, 0]

above = []
below = []
pp_0 = sorted(pp_0)
i = 0
for e in pp_0:
    if e < 0.5:
        above.append(e)
    else:
        below.append(e)

least_above = above[-2:]
least_below = below[:2]

df_with_names = pd.read_csv('data_with_names.csv', header=0)

print("The games that our model what the least sure about classifying as 0: ")
print(df_with_names['Name'][pp_0.index(least_below[0])], ', ', df_with_names['Name'][pp_0.index(least_below[1])])
print()

print("The games that our model what the least sure about classifying as 1: ")
print(df_with_names['Name'][pp_0.index(least_above[0])], ', ', df_with_names['Name'][pp_0.index(least_above[1])])



# ***********----------- SVM ------------******************

#----Defulat SVM Model----
model_svm = SVC(kernel='linear',random_state=42)
print(model_svm)
train_result_svm = pd.DataFrame()
valid_result_svm = pd.DataFrame()
predictions = pd.DataFrame()
for train_index , val_index in KF.split(X_train):
    train_result_md = pd.DataFrame(None)
    valid_result_md = pd.DataFrame(None)
    model_svm.fit(X_train.iloc[train_index] , Y_train.iloc[train_index])
    acc_train_svm = accuracy_score(Y_train.iloc[train_index] ,  model_svm.predict(X_train.iloc[train_index]))
    acc_val_svm = accuracy_score(Y_train.iloc[val_index],  model_svm.predict(X_train.iloc[val_index]))
    train_result_svm = train_result_svm.append({'Train accuracy':acc_train_svm},ignore_index=True)
    valid_result_svm = valid_result_svm.append({'Validation accuracy':acc_val_svm},ignore_index=True)
avg_acc_train_svm = train_result_svm.mean()
avg_acc_val_svm = valid_result_svm.mean()
print(avg_acc_train_svm)
print(avg_acc_val_svm)

#----Hyperparameters Tuning----
param_grid = {'C': [0.01,0.1,0.5,1,3,10,100]
             }
grid_search = GridSearchCV(estimator=SVC(kernel='linear',random_state=42), param_grid=param_grid, refit=True, cv=KF , return_train_score=True )
grid_search.fit(X_train,Y_train)
Results = pd.DataFrame(grid_search.cv_results_)
print('The best parameters are:' ,grid_search.best_params_)
y_val = Results['mean_test_score']
y_train = Results['mean_train_score']
results_grid_search1 = pd.DataFrame(Results).sort_values('rank_test_score')[['params', 'mean_test_score']]
results_grid_search2 = pd.DataFrame(Results).sort_values('mean_train_score',ascending=False)[['params', 'mean_train_score']]
headers_val = ["Number", "Parameters", "Validation score"]
headers_train = ["Number", "Parameters", "Train score"]
print(tabulate(results_grid_search1, headers = headers_val, tablefmt = "grid"))
print(tabulate(results_grid_search2, headers = headers_train, tablefmt = "grid"))
#plot accuracy for validation
plt.plot(y_val)
plt.ylabel('Validation accuracy', fontsize = 10)
plt.xlabel('Iteration', fontsize = 10)
plt.title('Accuracy for each Iteration', fontsize = 20)
plt.show()
#plot accuracy for train
plt.plot(y_train)
plt.ylabel('Train accuracy', fontsize = 10)
plt.xlabel('Iteration', fontsize = 10)
plt.title('Accuracy for each Iteration', fontsize = 20)
plt.show()

#----Best Linear SVM After Tuning----
model_bsv = SVC(kernel='linear',C=100,random_state=42)
train_result_bsv = pd.DataFrame()
valid_result_bsv = pd.DataFrame()
for train_index , val_index in KF.split(X_train):
    model_bsv.fit(X_train.iloc[train_index] , Y_train.iloc[train_index])
    acc_train_bsv = accuracy_score(Y_train.iloc[train_index] ,  model_bsv.predict(X_train.iloc[train_index])) #accuracy of the train
    acc_val_bsv = accuracy_score(Y_train.iloc[val_index],  model_bsv.predict(X_train.iloc[val_index])) #accuracy of the validation
    train_result_bsv = train_result_bsv.append({'Train Best accuracy':acc_train_bsv},ignore_index=True)
    valid_result_bsv = valid_result_bsv.append({'Validation Best accuracy':acc_val_bsv},ignore_index=True)
    print('The Weights of the Features are:')
    print("[General_Sales , Critic_Weight , User_Weight , Year_of_Release_0 , Year_of_Release_1 , Genre_Action ,\n Genre_Adventure , Genre_Fighting ,"
          " Genre_Misc , Genre_Platform , Genre_Puzzle , Genre_Racing ,\n Genre_Role-Playing , Genre_Shooter, Genre_Simulation , Genre_Sports , Genre_Strategy , Rating_E\n"
          ", Rating_E10+ , Rating_M , Rating_T , Platform_General_Microsoft_Xbox , Platform_General_Nintendo , Platform_General_PC\n , Platform_General_Sega ,Platform_General_Sony_Playstation ]")
    print(model_bsv.coef_)
    print('The intercept is:')
    print(model_bsv.intercept_)

avg_acc_train_bsv = train_result_bsv.mean()
avg_acc_val_bsv = valid_result_bsv.mean()
print(avg_acc_train_bsv)
print(avg_acc_val_bsv)


#----C influence on accuracy----
c_vec= [0.01,0.1,0.5,1,3,10,100]
result_bsv = pd.DataFrame()
train_result_bsv = pd.DataFrame()
valid_result_bsv = pd.DataFrame()
for c in c_vec:
    train_result_bsv = pd.DataFrame(None)
    valid_result_bsv = pd.DataFrame(None)
    for train_index, val_index in KF.split(X_train):
        model_bsv= SVC(kernel='linear',C=c,random_state=42) #Our best parameters after tuning!
        model_bsv.fit(X_train.iloc[train_index], Y_train.iloc[train_index])
        acc_train_bsv = accuracy_score(Y_train.iloc[train_index] ,  model_bsv.predict(X_train.iloc[train_index])) #accuracy of the train
        acc_val_bsv = accuracy_score(Y_train.iloc[val_index],  model_bsv.predict(X_train.iloc[val_index])) #accuracy of the validation
        train_result_bsv = train_result_bsv.append({'Train Best accuracy':acc_train_bsv},ignore_index=True)
        valid_result_bsv = valid_result_bsv.append({'Validation Best accuracy':acc_val_bsv},ignore_index=True)
    avg_train_bsv = train_result_bsv.mean()
    avg_val_bsv = valid_result_bsv.mean()
    result_bsv = result_bsv.append({'C': c,
                                    'train_acc': avg_train_bsv,
                                    'val_acc': avg_val_bsv},
                                        ignore_index=True)

plt.figure(figsize=(13, 4))
plt.plot(result_bsv['C'], result_bsv['train_acc'], marker='o', markersize=4)
plt.plot(result_bsv['C'], result_bsv['val_acc'], marker='o', markersize=4)
plt.title("Infulence of C on accuracy")
plt.ylabel('Accuracy', fontsize = 10)
plt.xlabel('C', fontsize = 10)
plt.legend(['Train accuracy', 'Validation accuracy'])
plt.show()

#
# ####___________ K-Means Clustering _________________#####
 # We will use the dame data as for the ANN
cluster_data = X_train_s
k = np.arange(1, 11, 1)
# for i in k:
#     kmeans = KMeans(n_clusters=k, max_iter=300, n_init=10, random_state=42)
#     kmeans.fit(cluster_data)
#     cluster_data['cluster'] = kmeans.predict(cluster_data)
#     print(cluster_data)

# Better way to to the algo
cluster_data = pd.DataFrame()
cluster_data['General_Sales'] = X_train_s['General_Sales']
cluster_data['Critic_Weight'] = X_train_s['Critic_Weight']
cluster_data['User_Weight'] = X_train_s['User_Weight']

pca = PCA(random_state=42, n_components= 2)
pca.fit(cluster_data)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
data_pca = pca.transform(cluster_data)
data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
data_pca['Y'] = Y_train
# sns.scatterplot(x='PC1', y='PC2', hue='Y', data=data_pca)
# plt.show()

data_pca.drop('Y', axis='columns', inplace=True)
print(data_pca)
kmeans = KMeans(n_clusters=4, max_iter=300, n_init=10, random_state=42)
kmeans.fit(data_pca)
data_pca['cluster'] = kmeans.predict(data_pca)
print(data_pca)

# sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=data_pca)
# plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], marker='+', s=100 ,color='red')
# plt.show()


#### finding the best K for the Kmeans model
data_pca.drop('cluster', axis='columns', inplace=True)
print(data_pca)
k = np.arange(3, 10, 1)
print(k)
dbs_list = []
silhouette_list = []
iner_list = []
for i in k:
    print("Results for k = ", i)
    kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10, random_state=42, init='k-means++')
    kmeans.fit(data_pca)
    data_pca['cluster'] = kmeans.predict(data_pca)
    clus_labels = kmeans.labels_
    dbs_list.append(davies_bouldin_score(data_pca, clus_labels))
    silhouette_list.append(silhouette_score(data_pca, clus_labels, metric='euclidean'))
    iner_list.append(kmeans.inertia_)
    # if i==6:
    #     sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=data_pca)
    #     plt.show()

db_score = {'Davies_Bouldin_Score':dbs_list, "Number of clusters": k}
j=3
for dbs in dbs_list:
    print('For k=',j, ', Davies-Bouldin score is:', dbs)
    j+=1

plt.plot(k, dbs_list, marker='o')
plt.title("Davies-bouldin")
plt.xlabel("Number of clusters")
plt.show()

j=3
for sil in silhouette_list:
    print('For k=',j, ', Silhouette score is:', sil)
    j+=1

plt.plot(k, silhouette_list, marker='o')
plt.title("Silhouette")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(k, iner_list, marker='o')
plt.title("Inertia")
plt.xlabel("Number of clusters")
plt.show()

# ## ____________ Implement on ANN model_______________
df_new_lab = pd.read_csv('XY_train_nonum.csv', header=0)
pd.options.display.max_columns = None
est = KBinsDiscretizer(n_bins=9, encode='ordinal', strategy='quantile')
df_new_lab['EU_Sales'] = est.fit_transform(df_new_lab[['EU_Sales']])
df_new_lab = df_new_lab.astype(
    dtype={'Year_of_Release': 'category', 'Genre': 'category', 'Rating': 'category', 'Platform_General': 'category',
           'General_Sales': 'float', 'Critic_Weight': 'float', 'User_Weight': 'float', 'EU_Sales': 'category'})

# ---- Split our Data to X_train and Y_train from the united csv ----
temp_df = df_new_lab.copy()
Y_train_newL = temp_df['EU_Sales']
temp_df.drop('EU_Sales', axis='columns', inplace=True)
X_train_newL = temp_df
X_train_newL = pd.get_dummies(X_train_newL)
# Scale
std_scaler.fit(X_train_newL)
X_train_newL['General_Sales'] = scaler.fit_transform(X_train_newL[['General_Sales']])

print(X_train_newL)
print(Y_train_newL)


ann_model = MLPClassifier(random_state=42, hidden_layer_sizes=(9, 9), max_iter=1000, solver='lbfgs')
ann_train_result = pd.DataFrame()
ann_valid_result = pd.DataFrame()
for train_idx, val_idx in KF.split(X_train_newL):
    x_train = X_train_newL.iloc[train_idx]
    y_train = Y_train_newL.iloc[train_idx]
    x_test = X_train_newL.iloc[val_idx]
    y_test = Y_train_newL.iloc[val_idx]

    ann_model.fit(x_train, y_train)
    acc_train = accuracy_score(y_train, ann_model.predict(x_train))
    acc_val = accuracy_score(y_test, ann_model.predict(x_test))
    ann_train_result = ann_train_result.append({'train_acc': acc_train}, ignore_index=True)
    ann_valid_result = ann_valid_result.append({'val_acc': acc_val}, ignore_index=True)

print(ann_train_result)
print(ann_valid_result)
avg_acc_train = ann_train_result.mean()
avg_acc_val = ann_valid_result.mean()
print(avg_acc_train)
print(avg_acc_val)


# #--------*********Evaluate between models*********--------------
precision_res = pd.DataFrame()
avg_fpr_svm = pd.DataFrame()
avg_tpr_svm = pd.DataFrame()
#---Precision for SVM---
model_bsv = SVC(kernel='linear',C=100,random_state=42)
avg_precision_svm = pd.DataFrame()
for train_index , val_index in KF.split(X_train):
    model_bsv.fit(X_train.iloc[train_index] , Y_train.iloc[train_index])
    y_score = model_bsv.predict(X_train.iloc[val_index])
    avg_precision = precision_score(Y_train.iloc[val_index] , y_score)
    avg_precision_svm = avg_precision_svm.append({'Precision SVM':avg_precision} , ignore_index=True)
    fpr_s,tpr_s,thresholds=roc_curve(Y_train.iloc[val_index] , y_score)
    avg_fpr_svm = avg_fpr_svm.append({'FPR':fpr_s.mean()} , ignore_index=True)
    avg_tpr_svm = avg_tpr_svm.append({'TPR':tpr_s.mean()}, ignore_index=True)
mean_svm=avg_precision_svm.mean()
print('FPR of SVM is: ' ,avg_fpr_svm.mean())
print('TPR of SVM is: ' ,avg_tpr_svm.mean())

#---Precision for DT---
model_dt = DecisionTreeClassifier(max_depth=6,ccp_alpha=0,criterion='gini',random_state=42)
avg_precision_dt = pd.DataFrame()
avg_fpr_dt = pd.DataFrame()
avg_tpr_dt = pd.DataFrame()
for train_index , val_index in KF.split(X_train):
    model_dt.fit(X_train.iloc[train_index] , Y_train.iloc[train_index])
    y_score = model_dt.predict(X_train.iloc[val_index])
    avg_precision = precision_score(Y_train.iloc[val_index] , y_score)
    avg_precision_dt = avg_precision_dt.append({'Precision DT':avg_precision} , ignore_index=True)
    fpr,tpr,thresholds=roc_curve(Y_train.iloc[val_index] , y_score)
    avg_fpr_dt = avg_fpr_dt.append({'FPR':fpr.mean()} , ignore_index=True)
    avg_tpr_dt = avg_tpr_dt.append({'TPR':tpr.mean()}, ignore_index=True)
mean_dt=avg_precision_dt.mean()
print('FPR of DT is: ' ,avg_fpr_dt.mean())
print('TPR of DT is: ' ,avg_tpr_dt.mean())
#
#
# #---Precision for ANN---
model_ANN = MLPClassifier(random_state=42, hidden_layer_sizes=(9,9), max_iter=1000, solver='lbfgs')
avg_precision_ANN = pd.DataFrame()
avg_fpr_mlp = pd.DataFrame()
avg_tpr_mlp = pd.DataFrame()
for train_index , val_index in KF.split(X_train):
    model_ANN.fit(X_train.iloc[train_index] , Y_train.iloc[train_index])
    y_score = model_ANN.predict(X_train.iloc[val_index])
    avg_precision = precision_score(Y_train.iloc[val_index] , y_score)
    avg_precision_ANN = avg_precision_ANN.append({'Precision ANN':avg_precision} , ignore_index=True)
    fpr_mlp,tpr_mlp,thresholds=roc_curve(Y_train.iloc[val_index] , y_score)
    avg_fpr_mlp = avg_fpr_dt.append({'FPR':fpr_mlp.mean()} , ignore_index=True)
    avg_tpr_mlp = avg_tpr_dt.append({'TPR':tpr_mlp.mean()}, ignore_index=True)
mean_ann=avg_precision_ANN.mean()
print('FPR of MLP is: ' ,avg_fpr_mlp.mean())
print('TPR of MLP is: ' ,avg_tpr_mlp.mean())

#print the result of precision
precision_res=precision_res.append({'SVM_res':mean_svm,'DT_res':mean_dt,'ann':mean_ann},ignore_index=True)
headers_val = ["SVM", "DT", "MLP"]
print(tabulate(precision_res, headers = headers_val, tablefmt = "grid",))

# ---Improvements---
# boosting
learning_rate_vec=[ 0.025,0.05,0.075,0.1,0.25, 0.5, 0.75,1]
n_estimators_vec = [40 ]
train_result_boost = pd.DataFrame()
valid_result_boost = pd.DataFrame()
res_b=pd.DataFrame()
for n in n_estimators_vec:
    for lr in learning_rate_vec:
        train_result_boost = pd.DataFrame(None)
        valid_result_boost = pd.DataFrame(None)
        for train_index, val_index in KF.split(X_train):
            model_boost = GradientBoostingClassifier(max_depth=9,n_estimators=n,learning_rate=lr,random_state=42)
            model_boost.fit(X_train.iloc[train_index], Y_train.iloc[train_index])
            acc_train_boost = accuracy_score(Y_train.iloc[train_index], model_boost.predict(X_train.iloc[train_index]))  # accuracy of the train
            acc_val_boost = accuracy_score(Y_train[val_index], model_boost.predict(X_train.iloc[val_index]))  # accuracy of the validation
            train_result_boost = train_result_boost.append({'Train accuracy': acc_train_boost}, ignore_index=True)
            valid_result_boost = valid_result_boost.append({'Validation accuracy': acc_val_boost}, ignore_index=True)
        avg_train_boost = train_result_boost.mean()
        avg_val_boost = valid_result_boost.mean()
        res_b=res_b.append({'n_est':n ,
                            'learning_rate':lr ,
                            'train':avg_train_boost,
                            'val': avg_val_boost},ignore_index=True)

headers_train = ['learning_rate','n_estimators' , 'Train Accuracy' , 'Validation Accuracy']
print(tabulate(res_b, headers = headers_train, tablefmt = "grid"))

#----Final predictions----
# --Read the X_test file and make it like our data--- did it and write it to csv file , then use the fixed file to the prediction!
df = pd.read_csv('X_test.csv', header=0)
pd.options.display.max_columns = None
df['Critic_Weight']=df['Critic_Score']*(np.sqrt(df['Critic_Count']))
df['User_Weight']=df['User_Score']*(np.sqrt(df['User_Count']))
def maximum_absolute_scaling(df):  # make the Critic_Score and User_Score in the same scale between 0-1.
    df_scaled = df.copy()  # copy the dataframe
    for column in df_scaled.columns:  # apply maximum absolute scaling
        if column == 'User_Weight' or column == 'Critic_Weight':
            df_scaled[column] = df_scaled[column] / df_scaled[column].abs().max()
    return df_scaled
# call the maximum_absolute_scaling function
df = maximum_absolute_scaling(df)

df.drop('Name', axis='columns', inplace=True)
df.drop('Publisher', axis='columns', inplace=True)
df.drop('Critic_Score', axis='columns', inplace=True)
df.drop('Critic_Count', axis='columns', inplace=True)
df.drop('User_Score', axis='columns', inplace=True)
df.drop('User_Count', axis='columns', inplace=True)
df.drop('Developer', axis='columns', inplace=True)
df.drop('Reviewed', axis='columns', inplace=True)

#----***Final prediction***----
df_p = pd.read_csv('X_test_fixed.csv', header=0)
pd.options.display.max_columns = None
X_test = df_p
X_test = pd.get_dummies(X_test)
model_p = GradientBoostingClassifier(max_depth=9,n_estimators=100,learning_rate=0.25,random_state=42)
model_p.fit(X_train, Y_train)
y_predictions =model_p.predict(X_test)
pd.DataFrame(y_predictions,columns=['y']).to_csv(r'C:\Users\עומר קידר\PycharmProjects\ML-PartB\y_test_prediction.csv')