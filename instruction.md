# MLOps_trnest_copy_skinCancerDataset_test
Ini adalah repositori untuk *submission* *MLOps* Proyek Terakhir. 
penjelasan bisa didapat dari *forum* dicoding https://www.dicoding.com/academies/443/discussions/215290
- Tahap awal - pembuatan akun railway && `CLI` instalasi</br>
Railway menyediakan free trial 5$ untuk 1 bulan untuk pendaftaran awal.</br>
Installasi railway dapat dilihat di[dokumen](https://docs.railway.app/guides/cli)-nya
    
- Tahap *Login* ke *railway*
```
railway login
```
tahap ini harus menggunakan *user* anda yang didaftarkan melalui *email* anda
- Tahap proyek inisiasi ke *railway* (Opsional bila anda **BELUM** pernah membuat)
```
railway init
```
Silahkan memberikan nama yang sesuai proyek (Saran : nama yang mudah diketik).

- Tahapan mengetahui proyek di railway (opsional bila **SUDAH** pernah membuat)
```
railway link
```
tahapan ini adalah tahapan untuk mengetahui railway dan proyek anda di local atau mengganti proyek lain

- Tahap pengiriman proyek ke server railway
```
railway up
```
ini adalah *command* proses pengiriman proyek dari local ke server dengan cepat. Bila berhasil proses deployment bisa dilihat :
> - di https://railway.app/dashboard
  - klik *tab-card* proyek yang anda ingin kerjakan
  - klik *map-card* di proyek anda telah deploy

- Tahap mengatur konfigurasi network (Terdapat versi perbedaan seiring waktu)
> - pada *tab setting* anda bisa mencari kata -Public Networking- </br>
  - lalu generate domain 
