import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier

#create dataframe
df = pd.read_csv(r'C:\Users\Daniel\Desktop\Git Projects\Collagen\Data\FTIR_merged_db_clean_2020-10-16.csv')


#delete unwanted columns
df.drop(['Specimen_ID', 'ORAU_Collagen', 'Site_Name', 'Site_Area', 'Site_Locus', 'Site_Basket', 'Site_Zone',
       'Site_Other_deposition', 'Taxon_Name', 'Taxon_Remarks', 'Element_Age', 'Element_General', 'Element_Specific',
      'Element_Unique', 'Element_Side', 'Element_Preservation', 'Element_information', 'Lab', 'Material', 'C14_Date',
       'Error_date'], axis=1, inplace=True)


#delete rows without available spectrum or Low or Indet collagen
L1 = []
L2 = []
for j in range(len(df.iloc[:, 0])):

    if df.Collagen[j] == 'Low': # delete rows with Indet or low collagen
        L1.append(j)
    elif df.Collagen[j] == 'Indet':
        L1.append(j)
    else:
        if df.FTIR[j] == 'No': # get list with the indexes of rows without spectrum
            L2.append(j)

L = L1 + L2

df.drop(L, axis=0, inplace=True)
df_original = df.copy(deep=True)

print('juasjuas')
print(df.iloc[:,2])

# Prepare X and Y
X = df.iloc[:, 3:]  # spectral data
y = df.iloc[:, 1]  # target

#reset indexes
new_index = pd.Series(range(0, X.shape[0], 1))
X.set_index([new_index], inplace=True)

# Divide data between training and test
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

X_o = X.copy(deep=True)
x_train_o = x_train.copy(deep=True)
x_test_o = x_test.copy(deep=True)
y_train_o = y_train.copy(deep=True)
y_test_o = y_test.copy(deep=True)
print(x_test_o)

# loop isolating spectral regions
spectrum_regions_repetition = []
Data_columns = 7467 # number of wavelength features
step_size = 100 # size of features groups
spectrum_regions = []
model_list = []
df_total = pd.DataFrame() #create empty dataframe to append dataframes of all sections
for i in range(1, int(Data_columns/step_size), 1): # iterate along 73 sections of spectrum

    #reset original data frames
    df = df_original.copy(deep=True)
    X = X_o.copy(deep=True)
    x_train = x_train_o.copy(deep=True)
    x_test = x_test_o.copy(deep=True)
    y_train = y_train_o.copy(deep=True)
    y_test = y_test_o.copy(deep=True)

    a = list(range(1, (1 + step_size*i)))
    b = list(range((1 + step_size*(i+1)), X.shape[1]))

    c = a + b
    # remove wavelengths we don´t want
    X.drop(X.iloc[:, c], axis=1, inplace=True)
    x_test.drop(x_test.iloc[:, c], axis=1, inplace=True)
    x_train.drop(x_train.iloc[:, c], axis=1, inplace=True)
    print('hier')
    print(X.shape)
    y_list = list(y)

    # standardizing the features
    X = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=3, svd_solver='auto')
    df_pca = pca.fit_transform(X=X)
    df_pca = pd.DataFrame(df_pca)

    df_pca.insert(3, column='Collagen', value=y_list)
    df_final = df_pca

    X = df_pca.iloc[:, :3]
    index_train = list(x_train_o.index)  # unfortunately the indexes of x
    index_test = list(x_test_o.index)
    x_train = X.iloc[index_train, :]
    x_test = X.iloc[index_test, :]
    y = df_pca.iloc[:, 3]

    # create a data frame with all data frames
    df_total.insert(loc=len(df_total.columns), column=str(i*3 - 2), value=list(df_pca.iloc[:, 0]))
    df_total.insert(loc=len(df_total.columns), column=str(i*3 - 1), value=list(df_pca.iloc[:, 1]))
    df_total.insert(loc=len(df_total.columns), column=str(i*3), value=list(df_pca.iloc[:, 2]))

    # Scale
    sc = StandardScaler().fit(X)
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # SVM Model
    print(x_train.shape) #!!!!!51 COLUMNS ARE BEING USED
    model = LinearSVC(C=1, loss='hinge')
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print('spectral section ' + str(i))
    print('SVM LinearSVC model accuracy ' + str(acc))
    model_list.append(model)
    spectrum_regions.append(acc)


# plot accuracy of every spectrum region
plt.figure(1)
x_axis = range(1, int(Data_columns/step_size), 1)
plt.grid(which='both')
plt.xlabel('spectrum section #')
plt.ylabel('accuracy of SVM_LinearSVC model for spectrum region')
plt.plot(x_axis, spectrum_regions)
plt.show()

#df = df_original.copy(deep=True) # data frame with the whole spectrum
print(df_total.shape)
print('hihihihi')
df = df_total
X = df_total.iloc[:, :]
y = df_original.iloc[:, 1]
#print(y.shape)


# reset original train and test data frames
index_train = list(x_train_o.index) # unfortunately the indexes of x
index_test = list(x_test_o.index)
x_train = df_total.iloc[index_train, :]
x_test = df_total.iloc[index_test, :]
y_train = y_train_o.copy(deep=True)
y_test = y_test_o.copy(deep=True)


Pred = []
# Predictions are done for every model
for index, m in enumerate(model_list):

    # Chose only columns corresponding to the right model
    Pred.append(list(m.predict(x_test.iloc[:, [3 * index, 3 * index + 1, 3 * index + 2]])))


#define weights
weights = []
df_acc = pd.read_csv(r'C:\Users\Daniel\Desktop\Git Projects\Collagen\acc_sections.csv')
#print(df_acc.iloc[:, 1])
for ind, s in enumerate(df_acc.iloc[:, 1]):
    if s > 0.75:
        weights.append(pow((s - 0.5) * 2, 5))
    else:
        weights.append(0)

Results = []
print(Pred)

def turn_into_number(x):
    if x == 'Yes':
        return 1
    else:
        return -1

# Transform predictions into 1, -1 nested lists
Pred_new = []
for l in range(len(Pred)):
    Pred_new.append(list(map(turn_into_number, Pred[l])))

print('miau')
print(len(Pred_new))
print(len(weights))

#Ponderate predictions with weights
Ponder = []
for q, v in enumerate(Pred_new):
    Ponder.append([x * float(weights[q]) for x in v])


T_Ponder = map(list, zip(*Ponder)) #transposes Pred
T_Ponder = list(T_Ponder)

# Voting of models weighted
for q in range(len(T_Ponder)):

    W = sum(T_Ponder[q])
    if W >= 0:
        W = 'Yes'
    else:
        W = 'No'
    #W = max(T_Pred[q], key=T_Pred.count) # Most frequent result
    Results.append(W)

print('Results: ')
print(Results)
y_test =list(y_test)
print('Reality: ')
print(y_test)

c = 0
for k, r in enumerate(Results):
    if r == y_test[k]:
        c += 1

print('accuracy is: ' + str(c/len(Results)))
print(df_total)
