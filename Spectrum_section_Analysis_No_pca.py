import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import tree

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

# delete spectral wavelength not relevant for study#
spectrum_regions_repetition = []
N = 100
Data_columns = 7467 # number of wavelength features
step_size = 3 # size of features groups
for j in range(0,N,1):
    spectrum_regions = []
    df = df_original.copy(deep=True)
    X = df.iloc[:, 3:]
    #print(df)
    #print(X)
    y = df.iloc[:, 1]
    #print(y)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # reset indexes
    train_index = pd.Series(range(0, x_train.shape[0], 1))
    x_train.set_index([train_index], inplace=True)
    #y_train.set_index([train_index], inplace=True)
    test_index = pd.Series(range(0, x_test.shape[0], 1))
    x_test.set_index([test_index], inplace=True)
    #y_test.set_index([test_index], inplace=True)

    X_o = X.copy(deep=True)
    x_train_o = x_train.copy(deep=True)
    x_test_o = x_test.copy(deep=True)
    y_train_o = y_train.copy(deep=True)
    y_test_o = y_test.copy(deep=True)
    for i in range(1, int(Data_columns/step_size), 1): # iterate along 74 sections of spectrum

        X_sec = X.iloc[:, [i,i-1+step_size]]
        x_train_sec = x_train.iloc[:, [i,i-1+step_size]]
        x_test_sec = x_test.iloc[:, [i,i-1+step_size]]


        # standardizing the features
        #X = StandardScaler().fit_transform(X)


        # Scale
        sc = StandardScaler().fit(X_sec)
        x_train_sec = sc.fit_transform(x_train_sec)
        x_test_sec = sc.transform(x_test_sec)

        # SVM Model
        #x_train = np.array(x_train)
        #x_test = np.array(x_test)


        #model = KNeighborsClassifier(n_neighbors=6)
        model = LinearSVC(C=1, loss='hinge')
        model.fit(x_train_sec, y_train)
        acc = model.score(x_test_sec, y_test)
        print(j, i)
        print('model accuracy ' + str(acc))
        spectrum_regions.append(acc)

    print(spectrum_regions)
    spectrum_regions_repetition.append(spectrum_regions)

matrix_acc = map(list, zip(*spectrum_regions_repetition)) #transposes spectrum_regions_repetition

mean_region_acc = [sum(x)/N for x in matrix_acc]

plt.figure(1)

x_axis = range(1, int(Data_columns/step_size), 1)


# create csv file with average accuracies
import csv
# field names
fields = ['Spectrum section', 'avg. accuracy']
columns = [x_axis, mean_region_acc]
rows = [[columns[i][j] for i in range(2)] for j in range(len(x_axis))]# transpose colums to get rows

filename = 'acc_3columns_svc_NOpca.csv'

# writing to csv file
with open(filename, 'w', newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile, delimiter=';')
    # writing the fields
    csvwriter.writerow(fields)
    # writing the data rows
    csvwriter.writerows(rows)


plt.grid(which='both')
plt.xlabel('spectrum section #')
plt.ylabel('mean accuracy of SVM_LinearSVC model for spectrum region')
plt.plot(x_axis, mean_region_acc)
plt.show()

