import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy import mean
from numpy import std
from scipy.stats import norm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from numpy import array
from sklearn.model_selection import KFold
from sklearn.svm import SVR

##########################################################################
# -----Read Data------
df = pd.read_csv('Xy_train.csv', header=0)
pd.options.display.max_columns = None
print(df.describe())  # summary table of the data

# ---- General functions ----

def print_hist(dataframe, str):
    plt.hist(dataframe[str], bins=50, density=True)
    plt.title(str + " histogram", fontsize=20)
    plt.xlabel(str, fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.show()


def eu_sales_scatter(dataframe, name):
    if not isinstance(name, str):
        print('wrong input')
        return
    plt.hist(dataframe[name], bins=50, density=True)
    plt.scatter(dataframe[name], dataframe['EU_Sales'])
    plt.title("EU_Sales by " + name, fontsize=20)
    plt.xlabel(name, fontsize=15)
    plt.ylabel('EU Sales', fontsize=15)
    plt.show()


def calc_dist(str):
    no_duplicates = df.drop_duplicates(subset=[str], keep='first')  # drop duplicates years
    no_duplicates_vec = sorted(no_duplicates[str])  # sort list
    sums = []
    for j in range(len(no_duplicates_vec)):  # create a list with the number of categories with value 0
        sums.insert(j, 0)
    i = 0
    for element in df[str]:  # counts how many times it appears in the data
        index = no_duplicates_vec.index(element)
        sums[index] += 1
    total = len(df.index)
    dist_list = []

    for e in range(len(no_duplicates_vec)):  # calc ratio
        dist_list.insert(e, sums[e] / total)
    d = {str: no_duplicates_vec, 'prob': dist_list}
    data_f = pd.DataFrame(d)
    return data_f


#################################################################
# *********-----Exploratory data analysis-----************


# ------------------------------------------------------------------------------------------------------------
# --Platform--
noDuplicatePlatform = df.drop_duplicates(subset=['Platform'], keep='first')  # drop duplicates platforms
print('the number of different platforms are', len(noDuplicatePlatform))
print('The platforms are: ', (sorted(noDuplicatePlatform['Platform'])))

# --Year of Release--
noDuplicateYears = df.drop_duplicates(subset=['Year_of_Release'], keep='first')  # drop duplicates years
print('the number of different years are', len(noDuplicateYears))
print('The years of release in the date are: ', sorted(noDuplicateYears['Year_of_Release']))

# --Genre--
noDuplicateGenre = df.drop_duplicates(subset=['Genre'], keep='first')  # drop duplicates Genres
print('the number of different genres are', len(noDuplicateGenre))
print('The genre are: ', (sorted(noDuplicateGenre['Genre'])))

# --Publisher--
noDuplicatePublisher = df.drop_duplicates(subset=['Publisher'], keep='first')  # drop duplicates publishers
print('the number of different publishers are', len(noDuplicatePublisher))
print('The publishers  are: ', (sorted(noDuplicatePublisher['Publisher'])))

# --Developer--
noDuplicateDeveloper = df.drop_duplicates(subset=['Developer'], keep='first')  # drop duplicates developers
print('the number of different developers are', len(noDuplicateDeveloper))
print('The developers are: ', (sorted(noDuplicateDeveloper['Developer'])))

# --Rating--
noDuplicateRating = df.drop_duplicates(subset=['Rating'], keep='first')  # drop duplicates ratings
print('the number of different ratings are', len(noDuplicateRating))
print('The ratings are: ', (sorted(noDuplicateRating['Rating'])))
# ------------------------------------------------------------------------------------------------------------

# --------box plots for continuous variables with the exceptions---------------

# ###--------NA_Sales----------
boxPlot=plt.axes()
sns.boxplot(y='NA_Sales', data=df)
boxPlot.set_title('NA Sales Boxplot')
plt.show()
#  #--------JP_Sales----------
boxPlot1=plt.axes()
sns.boxplot(y='JP_Sales', data=df)
boxPlot1.set_title('JP_Sales Boxplot')
plt.show()
#
# --------Other_Sales----------
boxPlot2=plt.axes()
sns.boxplot(y='Other_Sales', data=df)
boxPlot2.set_title('Other Sales Boxplot')
plt.show()
#
# --------Critic_Score----------
boxPlot3=plt.axes()
sns.boxplot(y='Critic_Score', data=df)
boxPlot3.set_title('Critic Score Boxplot')
plt.show()

# --------Critic_Count----------
boxPlot4=plt.axes()
sns.boxplot(y='Critic_Count', data=df)
boxPlot4.set_title('Critic Count Boxplot')
plt.show()


# --------User_Score----------
boxPlot5=plt.axes()
sns.boxplot(y='User_Score', data=df)
boxPlot5.set_title('User Score Boxplot')
plt.show()

#  #--------User_Count----------
boxPlot6=plt.axes()
sns.boxplot(y='User_Count', data=df)
boxPlot6.set_title('User Count Boxplot')
plt.show()

#  #--------Year_of_Release----------
boxPlot7=plt.axes()
sns.boxplot(y='Year_of_Release', data=df)
boxPlot7.set_title('Year of Release Boxplot')
plt.show()

#  #--------EU_Sales----------
boxPlot8=plt.axes()
sns.boxplot(y='EU_Sales', data=df)
boxPlot8.set_title('EU Sales Boxplot')
plt.show()

# ----------------histograms- continuous variables only------------

# ----- EU_Sales -----
plt.hist(df['EU_Sales'], bins=200, density=True)
plt.title("EU_Sales histogram", fontsize=20)
plt.xlabel('Amount of sales (millions)', fontsize=15)
plt.ylabel('Density', fontsize=15)
sns.distplot(df['EU_Sales'], hist=False, kde=True)
plt.show()
# ----- NA_Sales -----
plt.hist(df['NA_Sales'], bins=200, density=True)
plt.title("NA_Sales histogram", fontsize=20)
plt.xlabel('Amount of sales (millions)', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.show()
# ----- JP_Sales -----
plt.hist(df['JP_Sales'], bins=200, density=True)
plt.title("JP_Sales histogram", fontsize=20)
plt.xlabel('Amount of sales (millions)', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.show()
# ----- Other_Sales -----
plt.hist(df['Other_Sales'], bins=200, density=True)
plt.title("Other_Sales histogram", fontsize=20)
plt.xlabel('Amount of sales (millions)', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.show()
# ----- Critic_Score -----
plt.hist(df['Critic_Score'], bins=10, density=True)
plt.title("Critic_Score histogram", fontsize=20)
plt.xlabel('Score', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.show()
# ----- Critic_Count -----
plt.hist(df['Critic_Count'], bins=15, density=True)
plt.title("Critic_Count histogram", fontsize=20)
plt.xlabel('Number of critics who gave score', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.show()
# ----- User_Score -----
plt.hist(df['User_Score'], bins=10, density=True)
plt.title("User_Score histogram", fontsize=20)
plt.xlabel('Score', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.show()
# ----- User_Count -----
plt.hist(df['User_Count'], bins=200, density=True)
plt.title("User_Count histogram", fontsize=20)
plt.xlabel('Number of users who gave score', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.show()
# ----- Year_of_Release -----
plt.hist(df['Year_of_Release'], bins=100, density=True)
plt.title("Year_of_Release histogram", fontsize=20)
plt.xlabel('Year of Release', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.show()

# -----------apriori probabilities and histograms - categorical variables only---------
# ---Count plot for Platform----
sns.countplot(x='Platform', data=df, palette='Set1')
plt.title("Platform Countplot", fontsize=20)
plt.xlabel('Platform', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

#---Count plot for Genre----
sns.countplot(x='Genre', data=df, palette='Set1')
plt.title("Genre Countplot", fontsize=20)
plt.xlabel('Genre', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

#---Count plot for Rating----
sns.countplot(x='Rating', data=df, palette='Set1')
plt.title("Rating Countplot", fontsize=20)
plt.xlabel('Rating', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

for str in df:
    if str == 'Platform' or str == 'Year_of_Release' or str == 'Genre' or str == 'Publisher' or str == 'Developer' or \
            str == 'Rating' or str == 'Reviewed':
        data_f = calc_dist(str)
        print(data_f)
        # print_hist(df, str)


#####################################################################
# **********------------ Data set creation -----------****************

# ^^^^^^^^^^^^^^^ Pre Processing ^^^^^^^^^^^^^^^^^^^^ #
# ----Redundancy in the data: Check if there is duplicate in the data----
checkDuplicates = df.drop_duplicates(subset=None, keep='first')
print(checkDuplicates)

# ------  Missing values: Check for null values in the data ---- #
print('Number of null values is: ')
print(df.isnull().values.sum()) # there are no missing values so dont need to do anything.

# ----Exceptional values----
df = df.drop(df[df['Year_of_Release'] < 1990].index)  # remove data that release before 1990.
df = df.drop(df[df['Rating'] == 'K-A'].index)  # remove K-A rating from data.
df = df.drop(df[df['Rating'] == 'RP'].index)  # remove RP rating from data.
df = df.drop(df[df['Rating'] == 'AO'].index)  # remove AO rating from data.
print(df)

#----Cange Year of Release to 0/1 , 0- under 2008 , 1-above 2008
df['Year_of_Release']=(df['Year_of_Release']>2008).astype(int)

# ------ Data type conversions ----- #
# developer, publisher, rating, platform, genre
# ---Count plot for year of release----
sns.countplot(x='Year_of_Release', data=df, palette='Set1')
plt.title("Year of Release Countplot", fontsize=20)
plt.xlabel('0 - under 2008     1 - above 2008', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()


# Merging platforms:
# Copy and add one more platform column to the end of dataset
df['Platform_General'] = df['Platform']

# Convert console subnames to the general names
df.loc[df['Platform'] == 'PS3', 'Platform_General'] = 'Sony_Playstation'
df.loc[df['Platform'] == 'PS', 'Platform_General'] = 'Sony_Playstation'
df.loc[df['Platform'] == 'PS2', 'Platform_General'] = 'Sony_Playstation'
df.loc[df['Platform'] == 'PS4', 'Platform_General'] = 'Sony_Playstation'
df.loc[df['Platform'] == 'PSP', 'Platform_General'] = 'Sony_Playstation'
df.loc[df['Platform'] == 'PSV', 'Platform_General'] = 'Sony_Playstation'
df.loc[df['Platform'] == 'Wii', 'Platform_General'] = 'Nintendo'
df.loc[df['Platform'] == 'DS', 'Platform_General'] = 'Nintendo'
df.loc[df['Platform'] == 'GBA', 'Platform_General'] = 'Nintendo'
df.loc[df['Platform'] == '3DS', 'Platform_General'] = 'Nintendo'
df.loc[df['Platform'] == 'WiiU', 'Platform_General'] = 'Nintendo'
df.loc[df['Platform'] == 'NES', 'Platform_General'] = 'Nintendo'
df.loc[df['Platform'] == 'SNES', 'Platform_General'] = 'Nintendo'
df.loc[df['Platform'] == 'N64', 'Platform_General'] = 'Nintendo'
df.loc[df['Platform'] == 'GB', 'Platform_General'] = 'Nintendo'
df.loc[df['Platform'] == 'GC', 'Platform_General'] = 'Nintendo'
df.loc[df['Platform'] == 'X360', 'Platform_General'] = 'Microsoft_Xbox'
df.loc[df['Platform'] == 'XB', 'Platform_General'] = 'Microsoft_Xbox'
df.loc[df['Platform'] == 'XOne', 'Platform_General'] = 'Microsoft_Xbox'
df.loc[df['Platform'] == '2600', 'Platform_General'] = 'Atari'
df.loc[df['Platform'] == 'DC', 'Platform_General'] = 'Sega'
df.loc[df['Platform'] == 'SAT', 'Platform_General'] = 'Sega'
df.loc[df['Platform'] == 'GG', 'Platform_General'] = 'Sega'
df.loc[df['Platform'] == 'WS', 'Platform_General'] = 'Bandal'
df.loc[df['Platform'] == 'TG16', 'Platform_General'] = 'Nec'
df.loc[df['Platform'] == '3DO', 'Platform_General'] = 'Panasonic'
df.loc[df['Platform'] == 'PCFX', 'Platform_General'] = 'Nec'

# Check uniq values of Platform_General
df["Platform_General"].unique()

# ---Count plot for general platform----
sns.countplot(x='Platform_General', data=df, palette='Set1')
plt.title("Platform General Countplot", fontsize=20)
plt.xlabel('Platform_General', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

# Gives each category a number
def replace_to_cat(name):
    if isinstance(name, str):
        labels = df[name].astype('category').cat.categories.tolist()
        replace_map_comp = {name: {k: v for k, v in zip(labels, list(range(1, len(labels) + 1)))}}
        print(replace_map_comp)
        return replace_map_comp


cat_var = {'Developer', 'Publisher', 'Rating', 'Platform', 'Genre', 'Reviewed', 'Platform_General'}
# df_replace = df.copy()
# Replace in a new data frame all the categorical variables with numbers

for cv in cat_var:
    replace_map_comp = replace_to_cat(cv)
    df.replace(replace_map_comp, inplace=True)




# ^^^^^^^^^^ Feature Extraction ^^^^^^^^^^^
sales_sum_1 = df['NA_Sales'] + df['Other_Sales']
sales_sum_2 = sales_sum_1 + df['JP_Sales']
df['NA_Other_Sales'] = sales_sum_1
df['General_Sales'] = sales_sum_2
df['Critic_Weight']=df['Critic_Score']*(np.sqrt(df['Critic_Count']))
df['User_Weight']=df['User_Score']*(np.sqrt(df['User_Count']))
print(df)

# ^^^^^^^^^^ Feature Representation ^^^^^^^^^^^

def maximum_absolute_scaling(df):  # make the Critic_Score and User_Score in the same scale between 0-1.
    df_scaled = df.copy()  # copy the dataframe
    for column in df_scaled.columns:  # apply maximum absolute scaling
        if column == 'User_Weight' or column == 'Critic_Weight':
            df_scaled[column] = df_scaled[column] / df_scaled[column].abs().max()
    return df_scaled
# call the maximum_absolute_scaling function
df = maximum_absolute_scaling(df)
print (df)

#-----------Scatter plot for General sales and EU Sales ---------
dataFrame = pd.DataFrame(data=df, columns=['General_Sales','EU_Sales']);
dataFrame.plot.scatter(x='General_Sales', y='EU_Sales', title= "Scatter plot between General_Sales to EU_SALES");
plt.show(block=True);


# ^^^^^^^^^^ Feature Selection ^^^^^^^^^^^
df_new = df.copy()
df_new.drop('Platform', axis='columns', inplace=True)
df_new.drop('NA_Sales', axis='columns', inplace=True)
df_new.drop('JP_Sales', axis='columns', inplace=True)
df_new.drop('Other_Sales', axis='columns', inplace=True)
df_new.drop('Developer', axis='columns', inplace=True)
df_new.drop('Publisher', axis='columns', inplace=True)
df_new.drop('Reviewed', axis='columns', inplace=True)
df_new.drop('NA_Other_Sales', axis='columns', inplace=True)
df_new.drop('Name', axis='columns', inplace=True)
df_new.drop('Critic_Score', axis='columns', inplace=True)
df_new.drop('Critic_Count', axis='columns', inplace=True)
df_new.drop('User_Score', axis='columns', inplace=True)
df_new.drop('User_Count', axis='columns', inplace=True)


# no cat -> only for regression
df_no_cat = df_new.copy()
df_no_cat.drop('Year_of_Release', axis='columns', inplace=True)
df_no_cat.drop('Genre', axis='columns', inplace=True)
df_no_cat.drop('Rating', axis='columns', inplace=True)
df_no_cat.drop('Platform_General', axis='columns', inplace=True)
print(df_no_cat)

# cat -> only for chi^2
df_only_cat = df_new.copy()
df_only_cat.drop('General_Sales', axis='columns', inplace=True)
df_only_cat.drop('User_Weight', axis='columns', inplace=True)
df_only_cat.drop('Critic_Weight', axis='columns', inplace=True)


# # ---- Pearson correlation of continuous variables

print(df.corr(method='pearson')['EU_Sales'])
heatMap = plt.axes()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
heatMap.set_title(' Variables correlations')
plt.show()

# ---- Continuous variables selection using liner regression
def forward_selected(data, response):
    """Linear model designed by forward selection.
    response: string, name of response column in data
    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """

    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} ".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} ".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model


lm = forward_selected(df_no_cat,'EU_Sales')
print(lm.model.formula)
print(lm.rsquared_adj)
heatMap = plt.axes()
sns.heatmap(df_no_cat.corr(), annot=True, cmap='coolwarm')
heatMap.set_title(' Variables correlations')
plt.show()

# Categorical selection

plt.scatter(df_only_cat['Rating'], df_only_cat['EU_Sales'])
plt.title("EU_Sales by " + 'Rating', fontsize=20)
plt.xlabel("'E': 1, 'E10+': 2, 'M': 3, 'T': 4", fontsize=15)
plt.ylabel('EU Sales', fontsize=15)
plt.show()

plt.scatter(df_only_cat['Platform_General'], df_only_cat['EU_Sales'])
plt.title("EU_Sales by " + 'Platform_General', fontsize=20)
plt.xlabel("'Microsoft_Xbox': 1, 'Nintendo': 2, 'PC': 3, 'Sega': 4, 'Sony_Playstation': 5", fontsize=15)
plt.ylabel('EU Sales', fontsize=15)
plt.show()

plt.scatter(df_only_cat['Genre'], df_only_cat['EU_Sales'])
plt.title("EU_Sales by " + 'Genre', fontsize=20)
plt.xlabel("'Action': 1, 'Adventure': 2, 'Fighting': 3, 'Misc': 4, 'Platform': 5, 'Puzzle': 6, 'Racing': 7, 'Role-Playing': 8, 'Shooter': 9, 'Simulation': 10, 'Sports': 11, 'Strategy': 12}", fontsize=15)
plt.ylabel('EU Sales', fontsize=15)
plt.show()

plt.scatter(df_only_cat['Year_of_Release'], df_only_cat['EU_Sales'])
plt.title("EU_Sales by " + 'Year_of_Release', fontsize=20)
plt.xlabel("<2008: 0, >2008: 1", fontsize=15)
plt.ylabel('EU Sales', fontsize=15)
plt.show()

plt.scatter(df['Publisher'], df['EU_Sales'])
plt.title("EU_Sales by " + 'Publisher', fontsize=20)
plt.xlabel("Publisher", fontsize=15)
plt.ylabel('EU Sales', fontsize=15)
plt.show()

plt.scatter(df['Developer'], df['EU_Sales'])
plt.title("EU_Sales by " + 'Developer', fontsize=20)
plt.xlabel("Developer", fontsize=15)
plt.ylabel('EU Sales', fontsize=15)
plt.show()

# ^^^^^^^^^^ Dimensionality Reduction ^^^^^^^^^^^
#PCA
dfWithoutY=df_no_cat.drop(['EU_Sales'] , axis=1) #I want to make the PCA only on the continuous variables and without the Y.
dataScaled=preprocessing.scale(dfWithoutY)
pca=PCA(n_components=2) # I gave to the PCA 3 continuous variables and we want to reduce dimension , so we choose 2.
pca.fit(dataScaled)
pca_data=pca.transform(dataScaled)
var=np.round(pca.explained_variance_ratio_*100 , decimals=1)
labels = ['PC'+str(x) for x in range(1, len(var)+1)]
plt.bar(x=range(1,len(var)+1) ,height=var , tick_label=labels)
plt.ylabel('% of Explaind Variance')
plt.xlabel('Principal Component')
plt.title('PCA Plot')
plt.show()
PCi=pca.transform(dataScaled)
print(PCi)


############################################################################
# ***********----------- Model Training ------------******************

#----K-fold------
KF = KFold(n_splits=10)
for train, test in KF.split(df):
       print("%s %s" % (train, test))