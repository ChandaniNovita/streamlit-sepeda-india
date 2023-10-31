# Laporan Proyek Machine Learning
### Nama : Chandani Novitasari
### Nim :211351036
### Kelas : Malam A

## Domain Proyek

Estimasi harga sepeda di India ini bisa digunakan untuk memprediksi patokan harga untuk orang orang yang ingin berbelanja sepeda di India.


## Business Understanding

Dapat dilakukan secara onlline sehingga bisa menghemat waktu dan tenaga untuk mencari spesifikasi jenis sepeda dan harganya di India.


### Problem Statements

- Dalam dataset ini belum bisa digunakan untuk memprediksi harga sepeda sesuai spesifikasi yang diinginkan customer
- Tidak memiliki wawasan pada harga sepeda berdasarkan modelnya

### Goals

- Membuat penelitian untuk memprediksi harga sepeda di India
- Melakukan analisis pada data

    ### Solution statements
    - Memilih kolom yang tepat dan melakukan Regresi Linear
    - memilih kolom yang tepat dan metode yang tepat yakni meliputi pemilihan plot yang sesuai.

## Data Understanding
Kumpulan data ini berisi sekitar 8 ribu catatan harga sepeda bekas di India. Data telah dikumpulkan dari salah satu portal online terkemuka untuk menjual sepeda bekas di India melalui web scrapping. [Berikut adalah datasetnya](https://www.kaggle.com/datasets/ropali/used-bike-price-in-india/data)

### Variabel-variabel pada Harga Sepeda di India adalah sebagai berikut:
- model_year : Tahun pembuatan model[contoh;2017,2016,dll] [numerik]
- kms_driven : Total kilometer yang telah ditempuh sepeda.[contoh; 17000,50000,dll][Numerik]
- owner : Mewakili jenis pemilik sepeda tersebut seperti pemilik pertama yang berarti pemilik saat ini telah membeli sepeda tersebut dalam keadaan baru, pemilik kedua berarti sepeda tersebut telah dijual kepada pemilik ini dari pemilik pertama dan seterusnya.[contoh;first owner][Numerik]
- location : Lokasi penjual.[contoh; bangalor]
- mileage : Rata-rata jarak tempuh yang diberikan sepeda. Ini direpresentasikan sebagai kilometer per liter bensin (kmpl).[contoh; 35.00][Numerik]
- power : dalam satuan Bhp. BHP adalah laju penyaluran torsi yang dihasilkan oleh mesin sepeda ke roda. Semakin cepat deliveryability maka semakin tinggi kecepatan sepeda motor dan sebaliknya. Untuk sepeda yang memiliki BHP lebih rendah dapat menarik beban yang lebih tinggi dan untuk sepeda yang memiliki BHP lebih besar dapat mendorong sepeda dengan kecepatan lebih cepat.[contoh; 19.0, 19.8][Numerik]
- price : Ini adalah variabel target kumpulan data dalam rupee India[contoh;63500][Numerik]
- company : adalah nama perusahaan yang menjadi brand untuk sepeda.[contoh; Bajaj, Royal]
- model_bike : jenis-jenis model sepeda. [contoh; Avebger Cruise, Enfield Classic]
- age : usia penggunaan sepeda. [contoh; 7,5,6,11][Numerik]

## Data Preparation
##### Data Collection
Data set ini saya dapatkan dari website kaggle yang bernama Harga Sepeda di India, jika tertarik dengan dataset ini anda bisa klik link di atas.

##### Data Discovery and Profiling
Untuk bagian ini, kita akan menggunakan teknik EDA.
```phyton
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
```
Selanjutnya kita lakukan import files, dan upload file tersebut di google colab.
```
from google.colab import files
files.upload()
```
Setelah mengupload filenya, maka kita akan lanjut dengan membuat sebuah folder untuk menyimpan file kaggle.json yang sudah diupload tadi
```
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
Lanjut kita akan mendownload datasetnya 
```
!kaggle datasets download -d ropali/used-bike-price-in-india
```
Exstract file yang sudah di download
```
!mkdir used-bike-price-in-india
!unzip used-bike-price-in-india.zip -d used-bike-price-in-india
!ls used-bike-price-in-india
```
Memasukkan file csv yang telah diextract pada sebuah variable
```
data = pd.read_csv('/content/used-bike-price-in-india/bikes.csv')
```
Melihat 5 data paling atas di dalam dataset tersebut 
```
data.head()
```
Melihat jumlah baris dan kolom pada dataset
```
print('Number of rows:', data.shape[0])
print('Number of columns:', data.shape[1])
```
Untuk melihat mengenai type data dari masing-masing kolom kita bisa menggunakan property info,
```
data.info()
```
Gunakan perintah di bawah untuk melihat apakah ada data yang duplikat
```
data.duplicated().sum()
```
##### Data Preprocassing
Saya menghapus baris-baris ini karena jumlahnya sangat kecil dan menghapusnya tidak akan mempengaruhi kinerja model kita
```
data.dropna(inplace=True)
```
```
data.isnull().sum()
```
Mengekstrak kata pertama dan kedua dari 'model_name' dan menugaskannya ke kolom 'company' dan 'bike_model'
```
data['company'] = data['model_name'].apply(lambda x:' '.join(x.split()[0:1]))
```
```
data['bike_model'] = data['model_name'].apply(lambda x:' '.join(x.split()[1:3]))
```
Saya akan menambahkan kolom baru dan memberi label "age". Kolom ini akan menampilkan jumlah tahun sepeda tersebut telah dikendarai.
```
data['age'] = 2023-data['model_year']

```
Untuk mengekstrak nomor numerik, saya akan menghapus 'Km' dari kolom 'kms_driven'.
```
data['kms_driven'] = data['kms_driven'].str.replace('Km','')
```

Memfilter baris yang 'kms_driven' tidak berisi "Mil" dan menghapus baris dengan 'kms_driven' sebagai 'Yes'. Mengonversi kolom 'kms_driven' menjadi tipe integer.
```
data = data[~data['kms_driven'].str.contains("Mil")]
data = data[data['kms_driven'] != 'Yes ']
data['kms_driven'] = data['kms_driven'].astype(int)
```
Menunjukkan nilai unik pada kolom yang kursial dalam meprediksi harga
```
data['kms_driven'].unique()
```
```
data['owner'].unique()
```
```
data['location'].unique()
```
```
data['mileage'].unique()
```
```
data['power'].unique()
```
```
data['price'].unique()
```
##### Data Analysis
Company Counts
```
import plotly.express as px

df = data['company'].value_counts().reset_index()
df.columns = ['company', 'count']

fig = px.bar(df, x='company', y='count', color='company', title='Company Counts', text='count')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(xaxis_title='Company', yaxis_title='Count', showlegend=False)

fig.show()
```
![image](https://github.com/ChandaniNovita/streamlit-sepeda-india/assets/146313340/d2a67cbb-b822-4127-9187-34827e1ad382)

Company & Price
```
import plotly.express as px
fig = px.strip(data, x='company', y="price", orientation="h",color ='company')
fig.show()
```
![image](https://github.com/ChandaniNovita/streamlit-sepeda-india/blob/main/company%26price.png)
20 Model Sepeda Teratas berdasarkan Frekuensi
```
freq = data['bike_model'].value_counts()[:20]
```
```
fig = px.bar(freq, x=freq.index, y=freq.values, labels={'x': 'Bike Model', 'y': 'Frequency'},color='bike_model')
fig.update_layout(title='Top 20 Bike Models by Frequency', xaxis_tickangle=-45, height=400)

fig.show()
```
![image](https://github.com/ChandaniNovita/streamlit-sepeda-india/blob/main/topbike.png)
Frekuensi harga berdasarkan power
```
plt.figure(figsize=(20, 6))

colors = sns.color_palette('viridis', len(data['power'].unique()))

sns.lineplot(data=data, y='price', x='power', palette=colors)
plt.xticks(rotation=0)
plt.xlabel('Power')
plt.ylabel('Price')

plt.show()
```
Frekauensi harga berdasarkan pemilik
```

plt.figure(figsize=(8, 6))  # Increase figure size

#colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF']  # Specify the colors for the bars

sns.barplot(data=data, x=data['owner'], y='price', palette='plasma')
plt.xticks(rotation=45)  # Rotate x-axis labels
plt.xlabel('owner')  # Add x-axis label
plt.ylabel('Price')  # Add y-axis label
plt.title('Bar Plot')  # Add plot title

plt.show()
```
```
plt.figure(figsize=(20,6))
sns.heatmap(data.corr(),annot=True,cbar=False,cmap='winter')
plt.show()
```
## Modeling
Membuat salinan data 
```
data1 = data.copy()
```
Mengkonfersi data-data kategorikal dan string yang bersifat kategorikal menjadi numeric
```
encoder =LabelEncoder()
```
```
columns = ['model_name','owner','location','company','bike_model','model_year']
```
Menampilkan isi 5 baris pertama pada datasset yang sudah dikonfersi
```
for i in columns:
    data1[i] = encoder.fit_transform(data1[i])
```
```
data1.head()
```
Memasukkan kolom-kolom fitur yang ada di datasets dan juga kolom targetnya
```
features = ['model_year', 'kms_driven', 'owner', 'location', 'mileage', 'power','company', 'bike_model', 'age']
x = data1[features]
y = data1['price']
x.shape, y.shape
```
```
data1.drop(columns=['model_name'],inplace=True)
```
```
X= data1.drop(columns=['price'],axis=1)
y=data1['price']
```
Menentukan berapa persen dari dataset yang akan digunakan untuk test dan train, disini kita gunakan 30% untuk test dan sisanya untuk training alias 70%
```
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)
```
Mari kita lanjutkan dengan membuat model Regresi Liniernya
```
lr = LinearRegression()
```
```
lr.fit(x_train,y_train)
```
```
predict_lr = lr.predict(x_test)
```
```
score_lr = round((lr.score(x_test,y_test)*100),2)
print ("Model Score",score_lr,"%")
```
Hasilnya adalah : Model Score 73.66 %

Regresi Ridge
```
rid =Ridge(alpha=100)
```
```
rid.fit(x_train,y_train)
```
```
predic_rid = rid.predict(x_test)
```
```
score_rid = round((rid.score(x_test,y_test)*100),2)
print ("Model Score",score_rid,"%")
```
Model sudah selesai di buat, sekarang mari kita import menjadi file sav agar bisa digunakan pada projek streamlit
```phyton
import pickle

filename = 'bike_price_in_india.sav'
pickle.dump(lr,open(filename,'wb'))
```
## Evaluation
Disini saya menggunakan Root Mean Square Error atau (RMSE)
- Root Mean Square Error (RMSE) sendiri yaitu metode alternatif untuk mengevaluasi teknik peramalan yang digunakan untuk mengukur tingkat akurasi hasil perkiraan suatu model. Nilai yang dihasilkan RMSE merupakan nilai rata-rata kuadrat dari jumlah kesalahan pada model prediksi.
-Root Mean Squared Error (RMSE) merupakan salah satu cara untuk mengevaluasi model regresi linear dengan mengukur tingkat akurasi hasil perkiraan suatu model. RMSE dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan. RMSE tidak memiliki satuan.
```
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
tree_model = DecisionTreeRegressor()
tree_model.fit(x_train, y_train)
y_pred = tree_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"Root Mean Squared Error (RMSE): {rmse}")
```
Root Mean Squared Error (RMSE): 34261.42346797726

Nilai RMSE sebesar 33127.1884 menunjukkan bahwa rata-rata prediksi harga sepeda bekas dari model pohon keputusan memiliki error sekitar 33127.1884 Rupee India.
## Deployment
[Aplikasi Estimasi Harga Sepeda di India](https://app-estimasi-g9f35eay9vzsorffit963d.streamlit.app/)
![image](https://github.com/ChandaniNovita/streamlit-sepeda-india/blob/main/streamlit.png)

