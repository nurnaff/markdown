# Laporan Proyek Machine Learning - Nur Nafiiyah 
## Domain Proyek
Pada era revolusi industri 4.0 saat ini, produk elektronik dan digital semakin berkembang pesat. Perkembangan produk elektronik, terutama laptop, yang memiliki berbagai macam varian dan komponen berkualitas tinggi, membuat masyarakat ingin membelinya. Dalam hal ini, laptop adalah barang elektronik yang sangat diminati oleh masyarakat, terutama oleh pelajar dan mahasiswa. Masyarakat lebih suka memiliki laptop karena harganya yang semakin murah dan spesifikasinya yang bagus. Dengan laptop, dapat melakukan tugas-tugas yang biasanya membutuhkan peran teknologi informasi, seperti pengetikan, surat-menyurat, pengolahan data penjualan, proses belajar mengajar, dan lain-lain. Dalam memilih laptop dengan harga yang sesuai dengan kebutuhan masyarakat, terutama pelajar, mahasiswa, dan karyawan, masih bingung. Hal ini disebabkan oleh fakta bahwa pelanggan dihadapkan pada pemilihan beberapa spesifikasi laptop yang berbeda-beda dengan masing-masing keunggulan. Selain itu, pelanggan kurang memahami dan mendapatkan informasi yang cukup tentang keuntungan dan kekurangan dari setiap spesifikasi yang mereka pilih.

Proyek predictive analytics memiliki beberapa tahapan dalam proses pengembangannya. Keberhasilan proyek predictive analytics tidak hanya ditentukan oleh pemilihan algoritma machine learning saja, melainkan juga penerapan metodologi standar dalam mengelola seluruh tahapan atau siklus proyek. Cross-Industry Standard Process for Data Mining atau disingkat menjadi CRISP-DM merupakan salah satu metode standar proses analitik yang paling umum digunakan. Tahapan dari metode CRISP-DM dalam proses analitik dibagi menjadi enam fase utama, antara lain: Business understanding, Data understanding, Data preparation, Modeling, Evaluation.

## 1. Business Understanding

Tidak dapat diragukan lagi bahwa laptop sangat penting dalam kehidupan sehari-hari. Dalam kehidupan sehari-hari, manusia melakukan analisis dan penelitian sains. Seperti komputer, laptop selalu ada di kehidupan sehari-hari. Semua bidang termasuk perusahaan, bank, universitas mengubah statistik mentah menjadi informasi yang bermanfaat dan berguna. Komputer digunakan oleh insinyur, siswa, guru, perusahaan, dan badan usaha pemerintah untuk tugas khusus, hiburan, menghasilkan pendapatan, dan pekerjaan kantor. Laptop memudahkan dalam menyimpan arsip dan informasi penting. Laptop menyediakan perangkat lunak dan peralatan yang membuat pekerjaan administrasi lebih mudah dan produktif termasuk sistem pencarian, pemrosesan transaksi, dan perhitungan.

As [Mohammed Ali Shaik] berdasarkan artikel [link text](https://www.researchgate.net/publication/369955491_Laptop_Price_Prediction_using_Machine_Learning_Algorithms)

### 1.1 Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, proyek ini akan mengembangkan sebuah sistem prediksi harga laptop untuk menjawab permasalahan berikut.
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap harga laptop?
- Bagaimana hasil model machine learning dalam memprediksi harga laptop?

### 1.2 Goals
Untuk  menjawab pertanyaan tersebut, proyek ini akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:

- Mengetahui fitur yang paling berkorelasi dengan harga laptop.
- Membuat model machine learning yang dapat memprediksi harga laptop seakurat mungkin berdasarkan fitur-fitur yang ada.

### 1.3 Metodologi
Prediksi harga adalah tujuan yang ingin dicapai. Dalam predictive analytics, saat membuat prediksi variabel kontinu artinya menyelesaikan permasalahan regresi. Oleh karena itu, metodologi pada proyek ini adalah: membangun model regresi dengan harga laptop sebagai target.

### 1.4 Metrik
Metrik digunakan untuk mengevaluasi seberapa baik model dalam memprediksi harga. Proyek ini menggunakan metrik MAE dan MAPE.


## 2. Data Understanding

### Deskripsi Variabel

Terdapat 1275 baris dalam tabel harga laptop, dan 23 variabel atau kolom, kolom target sebagai output (y) adalah kolom harga laptop (Price_euros).

Daftar kolom dalam tabel harga laptop sebagai input (x): Company, Product, TypeName, Inches, Ram, OS, Weight, Screen, ScreenW, ScreenH, Touchscreen, IPSpanel, RetinaDisplay, CPU_company, CPU_freq, CPU_model, PrimaryStorage, SecondaryStorage, PrimaryStorageType, SecondaryStorageType,GPU_company, GPU_model.

Penjelasan kolom
Company: Produsen Laptop.
Product: Merek dan Model.
TypeName: Jenis Laptop (Notebook, Ultrabook, Gaming, …dll).
Inches: Ukuran Layar.
Ram: Jumlah total RAM di laptop (GB).
OS: Sistem Operasi yang terpasang.
Weight: Berat Laptop dalam kilogram.
Price_euros: Harga Laptop dalam Euro. (Target)
Screen: definisi layar (Standar, Full HD, 4K Ultra HD, Quad HD+).
ScreenW: lebar layar (piksel).
ScreenH: tinggi layar (piksel).
Touchscreen: apakah laptop memiliki layar sentuh atau tidak.
IPSpanel: apakah laptop memiliki IPSpanel atau tidak.
RetinaDisplay: apakah laptop memiliki layar retina atau tidak.
CPU_company CPU_freq: frekuensi CPU laptop (Hz).
CPU_model PrimaryStorage: ruang penyimpanan utama (GB).
PrimaryStorageType: jenis penyimpanan utama (HDD, SSD, Flash Storage, Hybrid).
SecondaryStorage: ruang penyimpanan sekunder jika ada (GB).
SecondaryStorageType: jenis penyimpanan sekunder (HDD, SSD, Hybrid, None).
GPU_company
GPU_model

Variabel yang mempunyai korelasi >0.2 terhadap harga laptop adalah: Ram (0,74), OS (0,29), Weight (0,21), ScreenW (0.55), ScreenH (0,55), IPSpanel (0,25), CPU_freq (0,43), CPU_model (0,47), SecondaryStorage (0,29), PrimaryStorageType (0,5), GPU_company (0,32).

Fungsi describe() memberikan informasi statistik pada masing-masing kolom yang bernilai float antara lain dari kolom Inches, Ram, Weight, Price_euros, ScreenW, ScreenH, CPU_freq, PrimaryStorage, SecondaryStorage.

Count  adalah jumlah sampel pada data sebanyak 1275 baris.
Mean adalah nilai rata-rata dari kolom.
Std adalah standar deviasi dari kolom.
Min yaitu nilai minimum setiap kolom.
Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
25% adalah kuartil pertama.
50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
75% adalah kuartil ketiga.
Max adalah nilai maksimum.

Kolom yang mempunyai tipe data angka (float atau int) adalah Inches, Ram, Weight, Price_euros, ScreenW, ScreenH, CPU_freq, PrimaryStorage, SecondaryStorage.

Kolom yang mempunyai tipe data teks (object) adalah Company, Product, TypeName, OS, Screen, Touchscreen, IPSpanel, RetinaDisplay, CPU_company, CPU_model, PrimaryStorageType, SecondaryStorageType, GPU_company, GPU_model.

Dataset diambil dari Kaggle dataset: https://www.kaggle.com/datasets/owm4096/laptop-prices

Membaca dataset yang telah terdownload

### Menampilkan ukuran data (banyak baris dan kolom)
Kode untuk mengecek ukuran data:
```
dt.shape
```
Hasil dari kode 
```dt.shape```
```
(1275, 23)
```

### Mengecek data ada yang kosong atau tidak
Kode untuk mengecek data berisi kosong
```
dt.isnull().sum()
```
Hasil kode 
```dt.isnull().sum()```

![1](https://github.com/user-attachments/assets/a2c32c94-9cb2-4ddf-97d2-a29886077526)
<br>
![2](https://github.com/user-attachments/assets/01c4166c-7002-4dbc-ae87-0f17b3b8e737)

### Mengecek data yang duplikat
Kode mengecek data yang duplikat
```
dt.duplicated().sum()
```
Hasil kode
```dt.duplicated().sum()```
```
0
```
## Data Preparation

### 1. Persiapan data (mengkonversi kolom yang kategori ke bentuk angka menggunakan LabelEncoder)
Total kolom tabel harga laptop ada 22 kolom/variabel, dan yang dilakukan konversi ke bentuk angka ada 14 kolom.
Proses melakukan konversi teks/kategori ke bentuk angka, kolom yang bertipe data teks adalah: Company, Product, TypeName, OS, Screen, Touchscreen, IPSpanel, RetinalDisplay, CPU_company, CPU_model, PrimaryStorageType, GPU_company, GPU_model dikonversi ke bentuk angka dengan perintal LabelEncoder.

### 2. Menentukan variabel x, dan y
Variabel input yang digunakan sebanyak 11 kolom dengan nilai korelasi >0.2 terhadap harga laptop: Ram (0,74), OS (0,29), Weight (0,21), ScreenW (0.55), ScreenH (0,55), IPSpanel (0,25), CPU_freq (0,43), CPU_model (0,47), SecondaryStorage (0,29), PrimaryStorageType (0,5), GPU_company (0,32).

### 3. Membagi data training 80% dan testing 20%
Jumlah data training ada 1020 baris, dan data tes 255 baris.

## Membuat model

Membuat model Random Forest untuk memprediksi harga laptop (berdasarkan artikel [link text](https://www.researchgate.net/publication/369955491_Laptop_Price_Prediction_using_Machine_Learning_Algorithms))

Algoritma random forest adalah salah satu algoritma supervised learning. Ia dapat digunakan untuk menyelesaikan masalah regresi. Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Apa itu model ensemble? Sederhananya, ia merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir.

Algoritma Random Forest membuat tree (pohon) dengan beberapa alternatif, dan cara menghasilkan prediksi data baru dari hasil pohon yang bagian daun (leaf) akan dihitung nilai rata-rata.

Variabel input yang digunakan sebanyak 11 kolom, yaitu: Ram, OS, Weight, ScreenW, ScreenH, IPSpanel, CPU_freq, CPU_model, SecondaryStorage, PrimaryStorageType, GPU_company.

Parameter dalam library RandomForestRegressor menggunakan default yang disediakan, yaitu:
- n_estimatorsint, default=100: Membuat alternatif pohon sebanyak 100.
- criterion{“squared_error”, “absolute_error”, “friedman_mse”, “poisson”}, default=”squared_error”: Melakukan proses split pemisahan atribut/fitur untuk menurunkan anaknya dengan menghitung error (mencari error terkecil).
- max_depthint, default=None: Membuat level anak sebanyak berapa level (tingkat kedalaman pohon).
- min_samples_splitint or float, default=2: Membagi jumlah data sampel untuk menghitung proses pemisahan atribut/fitur dalam pohon.
- min_samples_leafint or float, default=1: Membuat jumlah daun (leaf) di anak pohon minimal (default=1).


## Evaluasi
Jawaban dari problem statements

- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap harga laptop?

Variabel input yang mempunyai korelasi >0.2 terhadap harga laptop adalah: Ram (0,74), OS (0,29), Weight (0,21), ScreenW (0.55), ScreenH (0,55), IPSpanel (0,25), CPU_freq (0,43), CPU_model (0,47), SecondaryStorage (0,29), PrimaryStorageType (0,5), GPU_company (0,32). 
Artinya fitur yang dengan korelasi mendekati 1 mempunyai pengaruh yang tinggi terhadap penentuan harga laptop.

- Bagaimana hasil model machine learning dalam memprediksi harga laptop?

Hasil evaluasi model Random Forest dengan 11 kriteria/atribut/fitur di atas: MAE=179.48, dan MAPE 80.55%.
Semakin baik model adalah model dapat memprediksi yang persis seperti data aktual. Berarti jika selisih antara hasil prediksi dengan data aktual kecil maka model tersebut baik.
Sehingga model yang dibuat ini mempunyai kepetatan rata-rata dalam memprediksi harga laptop sebesar kurang lebih 80%.
