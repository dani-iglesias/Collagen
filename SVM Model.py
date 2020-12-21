import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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
df_indet = pd.DataFrame(columns=df.columns.values.tolist())
df_low = pd.DataFrame(columns=df.columns.values.tolist())
for j in range(len(df.iloc[:, 0])):

    if df.FTIR[j] == 'No':
        L2.append(j)
    else:
        if df.Collagen[j] == 'Low':  # delete rows with Indet or low collagen
            L1.append(j)
            df_low = df_low.append(df.iloc[j, :], ignore_index=True)
        elif df.Collagen[j] == 'Indet':
            L1.append(j)
            print(j)
            df_indet = df_indet.append(df.iloc[j, :], ignore_index=True)
            print(df_indet)
#    else:
#        if df.FTIR[j] == 'No':  # get list with the indexes of rows without spectrum
#            L2.append(j)

print('tool')
print(df_indet)
L = L1 + L2
df.drop(L, axis=0, inplace=True)
df_original = df.copy(deep=True)

# Remove wavelengths in which we are not interested
#a = list(range(2, 2454)) # 2367 (up to 1500 wavelength)
#b = list(range(2457, df.shape[1])) # 2543 (from 1630 wavelegth)
#c = a + b
#df.drop(df.iloc[:, c], axis=1, inplace=True)
print(df)

# Prepare X and Y
#X = df.loc[:, ['X1582.307', 'X1582.789', 'X1583.271']]  # spectral data
X = df.iloc[:, 1000:2000]
y = df.iloc[:, 1]  # column
print(X)
y_list = list(y)

# standardizing the features
X = StandardScaler().fit_transform(X)

# PCA
#pca = PCA(n_components=3, svd_solver='auto')
pca = LDA(n_components=1)
df_pca = pca.fit_transform(X, y)
df_pca = pd.DataFrame(df_pca)
df_pca.insert(1, column='Collagen', value=y_list)  # , axis=1)
df_final = df_pca

print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

X = df_pca.iloc[:, :1]
y = df_pca.iloc[:, 1]

# transform (with same transformation) Indet rows to later be tested
df_indet = df_indet.iloc[:, 1000:2000]
df_indet_pca = pd.DataFrame(pca.transform(df_indet))

df_low = df_low.iloc[:, 1000:2000]
df_low_pca = pd.DataFrame(pca.transform(df_low))

# Divide data between training and test
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

#Scale
#sc = StandardScaler().fit(X)
#x_train = sc.fit_transform(x_train)
#x_test = sc.transform(x_test)





clf1 = LinearSVC(C=1, loss='hinge')
clf2 = KNeighborsClassifier(n_neighbors=6)
clf3 = tree.DecisionTreeClassifier()

model = VotingClassifier(
    estimators=[('svc', clf1), ('knne', clf2), ('tr', clf3)],
    voting='hard'
)

clf1.fit(X, y)
#clf2.fit(x_train, y_train)
#clf3.fit(x_train, y_train)
#model.fit(x_train, y_train)
#acc1 = clf1.score(x_test, y_test)
#acc2 = clf2.score(x_test, y_test)
#acc3 = clf3.score(x_test, y_test)
#acc = model.score(x_test, y_test)
#print('The models accuracy is: ' + str(acc) + ', ' + str(acc1) + ', ' + str(acc2) + ', ' + str(acc3))

#print('dede')
print(clf1.predict(df_indet_pca))

plt.figure(1)

y = list(y)
print('df')
print(X)
X = X.iloc[:, 0]
print(X)
print(y)

X_indet = list(df_indet_pca.iloc[:, 0])
X_low = list(df_low_pca.iloc[:, 0])
X_yes = []
X_no = []
for i, a in enumerate(X):
    if y[i] == 'Yes':
        X_yes.append(a)
    if y[i] == 'No':
        X_no.append(a)



#X_yes = [x, i for x in X for i in y if i == 'Yes']
#X_no = [x for x in X for i in y if i == 'No']
#X_no = [x for [x, i] in [X, y] if i == 'No']
print(len(X_yes))
print(len(X_no))
print(len(X_indet))
#colors = ['r', 'b']
targets = ['Yes', 'No', 'Indet', 'Low']

plt.hist([X_yes, X_no, X_indet, X_low], color=['r', 'b', 'g', 'y'], bins=100)
#plt.hist(df_indet_pca, color='g', bins=100)
plt.legend(targets)

plt.show()
