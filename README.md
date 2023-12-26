# Holiday Packages Prediction using Machine Learning 🌟

<p align="center">
  <img src="https://github.com/riyouuyt/HolidayPackPred-Project/assets/122600889/e95b81b5-07b0-46d3-b370-81b79c7b4c52" alt="Holiday Packages predicition">
</p>

Proyek ini merupakan tugas akhir dari Kursus Data Science Batch 32 di Rakamin Academy. Tujuannya adalah menggunakan teknik-teknik machine learning untuk memprediksi apakah pelanggan akan membeli paket liburan baru yang akan diluncurkan oleh "Trips&Travel.com". Dengan menggunakan dataset yang komprehensif yang mencakup informasi pelanggan seperti usia, jenis kontak, durasi penawaran, pekerjaan, status perkawinan, dan berbagai variabel lainnya, proyek ini bertujuan untuk membangun model prediktif yang dapat mengidentifikasi pelanggan yang berpotensi melakukan pembelian.

Dengan hanya 19% pelanggan yang melakukan pembelian paket liburan pada tahun sebelumnya, proyek ini bertujuan untuk menyempurnakan penargetan pelanggan dan mengurangi biaya pemasaran yang tinggi dengan menerapkan model machine learning yang akurat untuk memprediksi perilaku pembelian pelanggan.

Proyek ini menunjukkan kemahiran dalam pra-pemrosesan data, analisis data eksploratif (EDA), teknik pengembangan fitur, pemilihan model, dan evaluasi. Hasil akhirnya adalah model machine learning yang kuat yang memberikan wawasan tentang perilaku pelanggan dan membantu dalam strategi pemasaran yang efektif untuk peluncuran paket liburan baru.

## Problem Statement 🎯
Hanya 19% dari total pelanggan yang melakukan pembelian paket liburan pada tahun sebelumnya. Hal ini mengindikasikan bahwa targeting pelanggan kurang tepat atau bahkan tidak dilakukan sama sekali, yang menyebabkan biaya pemasaran menjadi tinggi.

## Goal 🏆
Mencapai targeting pelanggan yang efektif berdasarkan karakteristik pelanggan untuk mengurangi biaya pemasaran.

## Objective 🚀
Membangun model machine learning yang mampu memprediksi pelanggan yang berpotensi membeli paket liburan, memanfaatkan data pelanggan yang tersedia di perusahaan.

## Business Metric 📊
Tingkat konversi: 19%

## Prerequisites 📋

**Download data Holiday-Packages-Prediction (Kaggle) [here](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction)**

## Dataset 📦
Dataset ini digunakan untuk memprediksi apakah customer akan mengambil packages baru yang akan diluncurkan oleh "Trips&Travel.com".

## **Data Understanding📊**

- `Customer ID` : Identifikasi unik yang diberikan kepada setiap pelanggan
- `ProdTaken` : Apakah pelanggan mengambil paket (**Ya**) atau tidak (**Tidak**)
- `Age` : Usia pelanggan
- `TypeOfContact` : Bagaimana pelanggan dihubungi (Diusulkan oleh Perusahaan atau Permintaan Sendiri)
- `CityTier` : Tingkat kota bergantung pada perkembangan kota, fasilitas penduduk, dan standar kehidupan. Kategorinya diurutkan yaitu...
- `DurationOfPitch` : Durasi penawaran oleh salesperson kepada pelanggan
- `Occupation` : Pekerjaan pelanggan
- `Gender` : Jenis kelamin, laki-laki atau perempuan
- `NumberOfPerson` : Total jumlah orang yang berencana melakukan perjalanan bersama pelanggan
- `NumberOfFollowup` : Jumlah tindak lanjut yang dilakukan oleh salesperson setelah penawaran
- `ProductPitched` : Produk yang ditawarkan oleh salesperson
- `PreferredProperty` : Penilaian properti hotel yang diinginkan oleh pelanggan (**3-5**)
- `MaritalStatus` : Status pernikahan pelanggan (**Menikah, Bercerai, Belum Menikah**)
- `NumberOfTrips` : Rata-rata jumlah perjalanan dalam setahun oleh pelanggan
- `Passport` : Memiliki paspor (**Ya**) atau tidak (**Tidak**)
- `PitchSatisfaction` : Skor kepuasan terhadap penawaran (**1-5**)
- `OwnCar` : Memiliki mobil (**Ya**) atau tidak (**Tidak**)
- `NumberOfChildren` : Jumlah anak dengan usia kurang dari 5 tahun yang berencana melakukan perjalanan bersama pelanggan
- `Designation` : Jabatan pelanggan di organisasi saat ini
- `MonthlyIncome` : Pendapatan bruto bulanan pelanggan


## EDA & Business Insights 📊🔍

Bagian ini akan membahas analisis eksploratif data (Exploratory Data Analysis/EDA) yang meliputi penelusuran dataset, visualisasi data, dan temuan yang relevan. Selain itu, di sini juga akan ditemukan wawasan bisnis yang mungkin seperti tren yang menarik, korelasi antar-fitur, atau pola yang dapat membantu dalam memahami perilaku pelanggan yang berpotensi membeli paket liburan. Ini menjadi landasan penting untuk memahami bagaimana model machine learning dapat diterapkan secara efektif pada data yang ada.

### Descriptive Statistics 📈
Statistik umum menunjukkan bahwa dataset terdiri dari 4888 entri dengan rata-rata usia pelanggan sebesar 37.62 tahun. Durasi penawaran rata-rata adalah 15.49 menit, dan rata-rata orang yang berencana melakukan perjalanan bersama pelanggan adalah 2.90 orang. Jumlah tindak lanjut oleh salesperson memiliki rata-rata sebanyak 3.71 kali, dengan rata-rata penilaian properti hotel yang diinginkan oleh pelanggan mencapai 3.58 bintang. Selain itu, rata-rata jumlah perjalanan dalam setahun oleh pelanggan adalah 3.24.

Data kategorikal menunjukkan distribusi sebagai berikut:

- **Jenis Kontak**: Terdapat 3444 data Self Enquiry dan 1419 data Company Invited.
- **Pekerjaan**: Mayoritas adalah Salaried dengan 2368 data, diikuti oleh Small Business (2084 data), Large Business (434 data), dan hanya 2 data dari Free Lancer.
- **Jenis Kelamin**: Jumlah pelanggan laki-laki lebih tinggi dengan 2916 data dibandingkan dengan pelanggan perempuan (1817 data) dan Fe Male (155 data).
- **Produk yang Ditawarkan**: Distribusi produk yang ditawarkan mencakup Basic (1842 data), Deluxe (1732 data), Standard (742 data), Super Deluxe (342 data), dan King (230 data).
- **Status Pernikahan**: Mayoritas pelanggan adalah yang sudah menikah (2340 data), diikuti oleh pelanggan yang bercerai (950 data), single (916 data), dan belum menikah (682 data).
- **Jabatan**: Mayoritas pelanggan memiliki jabatan Executive (1842 data) dan Manager (1732 data), dengan sejumlah kecil Senior Manager (742 data), AVP (342 data), dan VP (230 data).

### Univariate Analysis 📉

1. **Boxplot Analysis**

   ![image](https://github.com/riyouuyt/HolidayPackPred-Project/assets/122600889/89c296a1-6a27-49f9-96c2-adf534203691)

Observasi:
- Terlihat dari plot tersebut terlihat bahwa datanya terlihar normal dengan outlier yang tidak terlalu banyak dari seluruh kolom
- setiap kolom yang memiliki outlier akan kita bersihkan nanti di data cleaning process

2. **Histogram Analysis**

   ![image](https://github.com/riyouuyt/HolidayPackPred-Project/assets/122600889/889cd0fc-3774-4930-8a88-afb5d317745d)

Obserbasi:
- `Age` : Distribusi normal
- `DurationOfPitch` `NumberOfTips` `MonthlyIncome` : Positively Skewed
- `OwnCar` `Passport` `ProdTaken` : Bimodal
- `NumberOfPersonVisiting` `City Tier` `PreferredPropertyStar`: Trimodal
Data dengan positively skewed cenderung memiliki ekor panjang di sebelah kanan distribusi, yang memungkinkan adanya outlier, maka dari itu kita dapat mencoba beberapa transformasi misalnya log transform dan melakukan normalisasi data dengan melakukan Z-Score untuk mengurangi mean dan membaginya dengan standar deviasi. Normalisasi berguna dalam memastikan data yang dimiliki skala nya seragam


## Data Preprocessing 🛠️

### Data Cleansing 🧹

1. **Penanganan Missing Values**
   - Kolom `Age`, `TypeOfContact`, `DurationOfPitch`, `NumberOfFollowups`, `PreferredProperty`, `NumberOfTrips`, `NumberOfChildren`, dan `MonthlyIncome` memiliki nilai yang hilang.
   - Pengisian nilai hilang dilakukan sebagai berikut:
       - `Age`: Diisi dengan nilai rata-rata.
       - `TypeOfContact`: Diisi dengan nilai 'Self Enquiry' (modus).
       - `DurationOfPitch`: Diisi dengan nilai median.
       - `NumberOfFollowups`: Diisi dengan nilai median.
       - `PreferredProperty`: Diisi dengan nilai modus.
       - `NumberOfTrips`: Diisi dengan nilai median.
       - `NumberOfChildren`: Diisi dengan nilai modus.
       - `MonthlyIncome`: Diisi dengan nilai median.

2. **Penanganan Inconsistent Data**
   - Ditemukan inkonsistensi pada kolom `Gender` dan `MaritalStatus`.
   - Koreksi dilakukan dengan mengubah 'Fe Male' menjadi 'Female' pada kolom `Gender`.
   - 'Single' diperbaiki menjadi 'Unmarried' untuk menggambarkan kategori yang lebih tepat pada kolom `MaritalStatus`.
   - Konversi tipe data `Age` dari float menjadi integer.

3. **Penanganan Duplicate Data**
   - Setelah pemeriksaan mendalam, tidak ditemukan adanya data ganda dalam dataset yang kita miliki.



### Feature Engineering 🛠️

1. **Outlier Handling:**
Kami memulai dengan menangani outlier menggunakan metode z-score pada beberapa fitur seperti 'Usia (Age)', 'Durasi Penawaran (DurationOfPitch)', 'Jumlah Perjalanan (NumberOfTrips)', dan 'Pendapatan Bulanan (MonthlyIncome)'.

2. **Feature Transformation:**
Kami melakukan transformasi fitur dengan pendekatan yang berbeda. Fitur 'Durasi Penawaran' dan 'Pendapatan Bulanan' telah mengalami transformasi logaritma setelah dinormalisasi dan distandardisasi, menjadi 'DOP_norm', 'MI_std', dan 'NumberOfTrips'. Sementara itu, fitur 'Usia' telah kami normalisasi untuk memperoleh skala yang seragam.

3. **Data Splitting & Imbalance Handling:**
Data kami bagi dalam rasio 80:20 untuk melatih dan menguji model. Kami juga menangani ketidakseimbangan kelas menggunakan metode SMOTE untuk memastikan model tidak terlalu condong pada kelas mayoritas.

4. **Feature Encoding & Selection:**
Kami melakukan encoding pada fitur-fitur seperti 'Jenis Kontak', 'Jenis Kelamin', 'Produk yang Ditawarkan', dan 'Jabatan' menggunakan label encoding. Sementara itu, fitur 'Pekerjaan' dan 'Status Pernikahan' diencode dengan metode ONEHOT. Proses seleksi fitur dilakukan dengan uji Fclassif untuk fitur kategorikal dan uji ChiSquare untuk fitur numerikal.

## Model Comparison 🤖📊

### Model Performance 📈

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Decision Tree        | 0.90     | 0.95      | 0.92   | 0.94     |
| XGBoost              | 0.90     | 0.91      | 0.97   | 0.94     |
| Logistic Regression  | 0.83     | 0.88      | 0.91   | 0.90     |
| Random Forest        | 0.90     | 0.91      | 0.97   | 0.94     |
| Support Vector Machine | 0.85   | 0.88      | 0.94   | 0.91     |

### Model Comparison Summary 📊

Dengan mencoba 5 algoritma model di atas, nilai akurasi dan precision paling tinggi diperoleh oleh Decision Tree dengan akurasi 90% dan precision 95%.

### Decision Tree & Hyperparameter Tuning 🌳🔧

Hyperparameter tuning yang dilakukan menjadikan model decision tree tidak overfitting.

#### Before Tuning:
- Training Accuracy: 1.0
- Testing Accuracy: 0.89
- Precision: 0.95
- Recall: 0.92
- F1-Score: 0.94

#### After Tuning:
- Training Accuracy: 0.93
- Testing Accuracy: 0.88
- Precision: 0.93
- Recall: 0.92
- F1-Score: 0.93

### Confusion Matrix 📉

True Positive: 134
True Negative: 728
False Positive: 56
False Negative: 58

## Feature Importance 🌟🔍

Jumlah penghasilan berpengaruh terhadap keputusan customer untuk membeli package. Calon customer yang sudah memiliki paspor atau duration of pitch lebih lama memiliki probabilitas lebih tinggi untuk membeli package.

