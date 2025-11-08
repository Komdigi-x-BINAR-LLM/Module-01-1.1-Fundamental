# Implementasi model dan training sederhana menggunakan Pytorch

## Setup

Selain setup yang sudah dilakukan, praktikum ini perlu menginstall `torchvision` untuk memudahkan akses dataset Fashion MNIST.

```
# pip
pip install torchvision
# uv 
uv add torchvision
```

## Deskripsi
Kalian akan diberikan tugas untuk implementasi kode berikut:

Implementasi logistic regression sederhana untuk dataset iris
Implementasi CNN sederhana untuk dataset Fashion-MNIST

Untuk implementasi logistic regression, lakukan pada CPU.
untuk implementasi CNN sederhana, lakukan pada GPU. Bisa dijalankan pada google colab.

Untuk setiap implementasi, kalian harus implemen:
- implementasi dataloader termasuk membagi dataset minimal menjadi train dan test / val
- implementasi model menggunakan `Module`
- memilih loss function dan optimizer yang digunakan yang sesuai dengan task yang dilakukan
- implementasi training loop sederhana disertai menghitung validation loss
- implementasi metrik evaluasi
- serialisasi model

Selama implementasi, sangat disarankan mengikuti tutorial maupun membaca dokumentasi yang tersedia pada website pytorch.

Kalian akan diberikan dua template yang sudah memberikan library apa saja yang perlu digunakan serta memberi petunjuk apa saya yang perlu diimplementasi, lengkapi dan coba jalankan.

Untuk CNN, arsitekturnya dibebaskan, agar simpel disarankan menggunakan arsitektur kecil saja dengan 2 konvolusi lalu langsung dilanjutkan bagian fully connected

Ada kunci jawaban, tapi sangat disarankan mencoba sebaik mungkin terlebih dahulu sebelum melihat solusinya. Kunci jawaban hanya referensi saja dan tidak mencerminkan kalian harus melakukan apa secara tepatnya

### Bonus / side quest
- implementasi augmentasi menggunakan transforms pada dataloader
- implementasi mekanisme checkpointing, simpan weights yang mencapai performa terbaik
- 
