import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# create dataframe
df = pd.read_csv(r'C:\Users\Daniel\Desktop\Git Projects\Collagen\Data\FTIR_merged_db_clean_2020-10-16.csv')

# delete unwanted columns
df.drop(['Specimen_ID', 'ORAU_Collagen', 'Site_Name', 'Site_Area', 'Site_Locus', 'Site_Basket', 'Site_Zone',
         'Site_Other_deposition', 'Taxon_Name', 'Taxon_Remarks', 'Element_Age', 'Element_General', 'Element_Specific',
         'Element_Unique', 'Element_Side', 'Element_Preservation', 'Element_information', 'Lab', 'Material', 'C14_Date',
         'Error_date'], axis=1, inplace=True)

# delete rows without available spectrum or Low or Indet collagen
L1 = []
L2 = []
for j in range(len(df.iloc[:, 0])):

    if df.Collagen[j] == 'Low':  # delete rows with Indet or low collagen
        L1.append(j)
    elif df.Collagen[j] == 'Indet':
        L1.append(j)
    else:
        if df.FTIR[j] == 'No':  # get list with the indexes of rows without spectrum
            L2.append(j)

L = L1 + L2
df.drop(L, axis=0, inplace=True)
df_original = df.copy(deep=True)

# Remove wavelengths in which we are not interested
a = list(range(24, 2367)) # up to 1500 wavelength
b = list(range(2543, df.shape[1])) #from 1630 wavelegth
c = a + b
df.drop(df.iloc[:, c], axis=1, inplace=True)


# Prepare X and Y
X = df.iloc[:, 3:]  # spectral data
y = df.iloc[:, 1]  # column
y_list = list(y)

# standardizing the features
X = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=3, svd_solver='auto')
df_pca = pca.fit_transform(X=X)
df_pca = pd.DataFrame(df_pca)
df_pca.insert(3, column='Collagen', value=y_list)  # , axis=1)
df_final = df_pca

print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

X = df_pca.iloc[:, :3]
y = df_pca.iloc[:, 3]

# Divide data between training and test
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# KNN Model n_neighbors parameter optimization
accuracy = [0 for x in range(101)]
results = []
N = 500
for j in range(N): # iterations using different test and training data
    max_acc = 0
    N_sol = [] # array containing array with with array of n_neighbors giving the
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    sc = StandardScaler().fit(X)
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    for i in range(1, 101, 1): #testing from n_neighbors = 1 to 100

        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        accuracy[i] += acc
        if acc == max_acc:
              N_sol.append(i)
        elif acc > max_acc:
              max_acc = acc
              N_sol = []
              N_sol.append(i)
    results.append(N_sol) # creates array of arrays
    print(N_sol)
    print(max_acc)

final = []
for j in range(len(results)): # converts 2D list into 1D list
    [final.append(x) for x in results[j]]

av_acc = [(x/N)*100 for x in accuracy] #average accuracy(%) of n_neighbors parameter

plt.figure(1)
plt.ylabel("solution incidence")
plt.xlabel("n_neighbors")
plt.grid(which='both')
plt.xlim(0, 100)
plt.plot(range(101), av_acc)
plt.hist(final, bins=100)
plt.show()


'''
# Mean accuracy of KNN Model with n_neighbors=6
total = 0
N = 500
for i in range(N):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    sc = StandardScaler().fit(X)
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = KNeighborsClassifier(n_neighbors=6)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print('KNN model accuracy :' + str(acc))
    total += acc
mean_acc = total/N
print('average accuracy')
print(mean_acc)
'''


