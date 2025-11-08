# Hands on Pytorch

## Petunjuk set up environment python

### Menggunakan pip
#### Setup
```
python -m venv env
./env/scripts/activate.bat # windows
./env/scripts/activate # linux / mac
pip install torch numpy scikit-learn
```
#### Untuk menjalankan file, misal `hello_world.py`
```
# pastikan env aktif dulu kalau belum aktif
./env/scripts/activate.bat # windows
./env/scripts/activate # linux / mac

# jalankan
python hello_world.py
```

### Menggunakan uv
#### Setup
```
uv init
uv add torch numpy scikit-learn
```
#### Untuk menjalankan file, misal `hello_world.py`
```
uv run hello_world.py
```

## Daftar file
- [hello_world.py](hello_world.py) - Hello world Pytorch
- [important_attributes.py](important_attributes.py) - Mengenal atribut penting pytorch
- [tensor_init.py](tensor_init.py) - Mengenal beberapa metode inisialisasi tensor
- [logistic_regression.py](logistic_regression.py) - Implementasi regresi logistik menggunakan operasi pytorch
- [logistic_regression_torch_nn.py](logistic_regression_torch_nn.py) - Implementasi regresi logistik menggunakan operasi pytorch dalam modul `torch.nn`
- [module_example.py](module_example.py) - Implementasi operasi custom dan model menggunakan abstraksi `Module` dalam Pytorch
- [loss_and_backprop.py](loss_and_backprop.py) - Implementasi loss function dan menghitung gradien untuk proses training
- [serialize_deserialize.py](serialize_deserialize.py) - Contoh menyimpan dan membaca parameter model dan informasi lainnya
## Tugas

Cek folder [tugas-modul-1](tugas-modul-1)