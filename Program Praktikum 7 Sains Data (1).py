import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path_titanic = "https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv"

df_titanic = pd.read_csv(path_titanic)
#==============================================================================
#MENGATASI MISSING VALUE
##Mengecek Missing Value
df_titanic.isna().sum()
df_titanic.count()
df_titanic.dtypes

##Mengganti Nilai Missing Value
df_titanic_new = df_titanic.copy()
df_titanic_new["Age"].fillna(value = df_titanic_new["Age"].median(),
                             axis = 0, inplace= True)
df_titanic_new["Cabin"].fillna(value = "Unknown",
                               axis = 0, inplace= True)
df_titanic_new["Embarked"].fillna(value = "Unknown",
                                  axis = 0, inplace= True)
df_titanic_new.isna().sum()

##Menghapus Nilai Missing Value
df_titanic_test = df_titanic.copy()
df_titanic_test.dropna(subset=["Age"], inplace=True)
df_titanic_test.isna().sum()
#==============================================================================
#MENGATASI NILAI OUTLIERS
##Melakukan Visualisasi untuk Mengecek
df_titanic_new.boxplot(column = "Age")
plt.title("Boxplot Age")
plt.ylabel("Usia")
plt.show()

##Membuat Fungsi Mengecek
def cek_outliers(dataset,kolom: str):
    q1 = dataset[kolom].quantile(0.25)
    q3 = dataset[kolom].quantile(0.75)
    iqr = q3 - q1
    
    ba = q3 + (1.5*iqr)
    bb = q1 - (1.5*iqr)
    outliers = dataset[(dataset[kolom] < bb) | (dataset[kolom] > ba)]
    
    print(f"Batas Bawah: {bb}")
    print(f"Batas Atas: {ba}")
    print(f"Jumlah outliers: {len(outliers)}")
    return bb, ba

kolom_num = df_titanic_new.select_dtypes(include=["int64", "float64"]).columns
for col in kolom_num:
    print(f"\n=== Nilai Outliers Kolom {col} ===")
    cek_outliers(df_titanic_new, col)

##Mengganti Nilai Outliers dengan BA dan BB
bb, ba = cek_outliers(df_titanic_new, "Fare")
df_titanic_new["Fare"] = df_titanic_new["Fare"].clip(lower=max(bb,0), upper=ba)
cek_outliers(df_titanic_new, "Fare")
df_titanic_new["Fare"].describe()

##Mengganti Nilai Outliers dengan Median
df_titanic_test = df_titanic.copy()
bb_test, ba_test = cek_outliers(df_titanic_test, "Fare")
bukan_outlier = df_titanic_test["Fare"].between(bb_test, ba_test)
median_bersih = df_titanic_test.loc[bukan_outlier, "Fare"].median()

df_titanic_test.loc[~bukan_outlier, "Fare"] = median_bersih
cek_outliers_median = ((df_titanic_test["Fare"] < bb_test) | 
                       (df_titanic_test["Fare"] > ba_test)).sum()
print("Outlier", cek_outliers_median)

##Menghapus Nilai Outliers
df_titanic_test = df_titanic.copy()
bb_test, ba_test = cek_outliers(df_titanic_test, "Fare")
bukan_outlier = df_titanic_test["Fare"].between(bb_test, ba_test)
df_titanic_test_bersih = df_titanic_test.loc[bukan_outlier]
cek_outliers2 = ((df_titanic_test_bersih["Fare"] < bb_test) | 
                (df_titanic_test_bersih["Fare"] > ba_test)).sum()
print("Outlier Setelah Dibersihkan:", cek_outliers2)
    
#===============================================================================
#MELAKUKAN NORMALISASI DATA
norm_collist = ["Age", "Fare", "SibSp", "Parch"]

##Standarisasi
scaler = StandardScaler()
df_titanic_new[norm_collist] = scaler.fit_transform(df_titanic_new[norm_collist])
print(df_titanic_new[norm_collist].describe().loc[["mean", "std"]].round(3))

##MInMax Scale
df_titanic_test = df_titanic.copy()
scale_minmax = MinMaxScaler()
df_titanic_test[norm_collist] = scale_minmax.fit_transform(df_titanic_test[norm_collist])
print(df_titanic_test[norm_collist].describe())
#===============================================================================
#MENGATASI DATA DUPLIKAT
df_dup = pd.concat([df_titanic, df_titanic.sample(15, random_state=42)], ignore_index=True)
##Mengecek Duplikat
df_dup.duplicated().sum()
##Menghapus Duplikat
df_dup = df_dup.drop_duplicates().reset_index(drop=True)
df_dup.duplicated().sum()
#===============================================================================
#MELAKUKAN KONVERSI TIPE DATA DAN ENCODING DATA
##Mengecek Tipe Data
print(df_titanic_new.dtypes)
 
##Konversi Tipe Data
df_titanic_new["Sex"] = df_titanic_new["Sex"].astype("category")
df_titanic_new["Embarked"] = df_titanic_new["Embarked"].astype("category")
df_titanic_new["Pclass"] = pd.Categorical(df_titanic_new["Pclass"],
                                          categories=[1, 2, 3], ordered=True)
print(df_titanic_new.dtypes)
 
##Label Encoding (kategori 2 nilai): Sex
df_titanic_new["Sex"] = (df_titanic_new["Sex"] == "female").astype(int)
 
##Ordinal Encoding (kategori berurutan): Pclass
df_titanic_new["Pclass"] = df_titanic_new["Pclass"].cat.codes
 
##One-Hot Encoding (kategori tanpa urutan): Embarked
df_titanic_new = pd.get_dummies(df_titanic_new, columns=["Embarked"],
                                drop_first=True, dtype=int)
 
##Memeriksa Hasil
print(df_titanic_new.head())
print(df_titanic_new.dtypes)