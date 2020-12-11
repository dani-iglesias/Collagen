import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import tree
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

# frequency axis
freq_axis = df.iloc[:0, 3:]
freq_axis =list(map(lambda x: x[1:], freq_axis))
freq_axis = [float(x) for x in freq_axis]
print(freq_axis)
print(len(freq_axis))


print('juasjuas')
print(df.iloc[:,2])

# delete spectral wavelength not relevant for study#
spectrum_regions_repetition = []
N = 100
Data_columns = 7467 # number of wavelength features
step_size = 3 # size of features groups
for j in range(0,N,1):
    spectrum_regions = []
    for i in range(0, int(Data_columns/step_size), 1): # iterate along 74 sections of spectrum
        df = df_original.copy(deep=True)
        a = list(range(3, (3 + step_size*i)))
        b = list(range((3 + step_size*(i+1)), df.shape[1]))

        c = a + b
        df.drop(df.iloc[:, c], axis=1, inplace=True) #remove wavelengths we don´t want


        # Prepare X and Y
        X = df.iloc[:, 3:] # spectral data
        y = df.iloc[:, 1] # target
        y_list = list(y)

        # standardizing the features
        X = StandardScaler().fit_transform(X)

        # PCA
        pca = PCA(n_components=3, svd_solver='auto')
        df_pca = pca.fit_transform(X=X)
        df_pca = pd.DataFrame(df_pca)

        df_pca.insert(3, column='Collagen', value=y_list)
        df_final = df_pca

        #print(pca.explained_variance_ratio_)
        #print(sum(pca.explained_variance_ratio_))

        X = df_pca.iloc[:, :3]
        y = df_pca.iloc[:, 3]

        #Divide data between training and test

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


        # Scale
        sc = StandardScaler().fit(X)
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        # SVM Model

        #model = LinearSVC(C=1, loss='hinge')
        #model = KNeighborsClassifier(n_neighbors=6)
        #model = tree.DecisionTreeClassifier()

        clf1 = LinearSVC(C=1, loss='hinge')
        clf2 = KNeighborsClassifier(n_neighbors=6)
        clf3 = tree.DecisionTreeClassifier()

        model = VotingClassifier(
            estimators=[('svc', clf1), ('knne', clf2), ('tr', clf3)],
            voting='hard'
        )

        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        print(i, j)
        #print('spectral section ' + str(i))
        print('model accuracy ' + str(acc))
        spectrum_regions.append(acc)

    print(spectrum_regions)
    spectrum_regions_repetition.append(spectrum_regions)

matrix_acc = map(list, zip(*spectrum_regions_repetition))

mean_region_acc = [sum(x)/N for x in matrix_acc]

#x_axis = range(1, int(Data_columns/step_size), 1)
Positions = range(0, len(freq_axis) - Data_columns % step_size , step_size)
print(list(Positions))
x_axis = [freq_axis[i] for i in Positions]

print(len(x_axis))
print(len(mean_region_acc))
print('aquiii')

# create csv file with average accuracies
import csv
# field names
fields = ['Spectrum section', 'avg. accuracy']
columns = [x_axis, mean_region_acc]
rows = [[columns[i][j] for i in range(2)] for j in range(len(x_axis))]# transpose columns to get rows

# writing to csv file
filename = 'Mean accuracy of Voting_classifier pca_3columns.csv'
with open(filename, 'w', newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile, delimiter=';')
    # writing the fields
    csvwriter.writerow(fields)
    # writing the data rows
    csvwriter.writerows(rows)

plt.figure(1)

plt.grid(which='both')
plt.title(filename)
plt.xlabel('Wavelenght (nm)')
plt.ylabel('mean accuracy of model')
plt.plot(x_axis, mean_region_acc)
plt.show()

