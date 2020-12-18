import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

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
for j in range(len(df.iloc[:])):

    #if df.Collagen[j] == 'Low': # delete rows with Indet or low collagen
    #    L1.append(j)
    if df.Collagen[j] == 'Indet':
        L1.append(j)
    else:
        if df.FTIR[j] == 'No': # get list with the indexes of rows without spectrum
            L2.append(j)

L = L1 + L2

df.drop(L, axis=0, inplace=True)


# delete spectral wavelength not relevant for study#
a = list(range(24, 2367)) # up to 1500 wavelength
b = list(range(2543, df.shape[1])) #from 1630 wavelegth
#a = list(range(24, 2767))
#b = list(range(2943, df.shape[1]))

c = a + b
#df.drop(df.iloc[:, c], axis=1, inplace=True) #remove wavelengths outside the range
print(df)

#Plot spectra with and without Collagen

x = range(len(df.iloc[1, 3:])) #only takes colums corresponding to spectrum (3rd column)
itera = range(len(df.iloc[:]))
print(x)
print('ooooo')

for i in itera: #row

    if df.FTIR.iloc[i] == 'Yes':

        if df.Collagen.iloc[i] == 'Yes':

            y = df.iloc[i, 3:]
            plt.figure(1)
            plt.plot(x,y)

        if df.Collagen.iloc[i] == 'No':

            y = df.iloc[i, 3:]
            plt.figure(2)
            plt.plot(x, y)


# Prepare X and Y

X = df.iloc[:, 3:] # spectral data
y = df.iloc[:, 1] #column
y_list = list(y)

# standardizing the features

X = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=3, svd_solver='auto')
df_pca = pca.fit_transform(X=X)
df_pca = pd.DataFrame(df_pca)

df_pca.insert(3, column='Collagen', value=y_list) #, axis=1)
df_final = df_pca


print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
print(df_pca)

# plot scatter principal components
plt.figure(3)
ax = plt.axes(projection ="3d")

targets = ['Yes', 'No', 'Low']
colors = ['r', 'b', 'y']
for target, color in zip(targets, colors):
    indicesToKeep = df_final['Collagen'] == target
    ax.scatter3D(df_final.loc[indicesToKeep, 0], df_final.loc[indicesToKeep, 1], df_final.loc[indicesToKeep, 2], c=color)
    #plt.scatter(df_final.loc[indicesToKeep, 0], df_final.loc[indicesToKeep, 1], c=color)
plt.legend(targets)

#sns.scatterplot(x=df_final[:, 0], y=df_final[:, 1], s=30, hue=df_final[:, 2], palette=['green', 'blue'])

plt.show()

