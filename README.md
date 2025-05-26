Berdasarkan: [https://github.com/helme/ecg_ptbxl_benchmarking]

## File utama
* [mycode.ipynb](./mycode.ipynb) - Alur demo program utama
* [mycode-resnet.ipynb](./mycode-resnet.ipynb) - Training dan inference ResNet
* [models/inception1d.py](./models/inception1d.py) - Model Inception1D yang digunakan
* [models/xai_models.py](./models/xai_models.py) - Terdapat implementasi Grad-CAM serta Inception1D yang telah dimodifikasi dengan Grad-CAM.
* [models/resnet1d_chen.py](./models/resnet1d_chen.py) - Implementasi ResNet berdasarkan [Chen et al. 2021](https://doi.org/10.3389/fcvm.2021.654515)
* [dataloader.py](./dataloader.py) - Program untuk loading PTB-XL menggunakan PyTorch Dataloader

## File helper
* [ecg_processing.py](./ecg_processing.py) - Terdapat fungsi-fungsi pemrosesan dan visualisasi sinyal EKG
* [utils.py](./utils.py) - Beberapa fungsi prapemrosesan dan evaluasi (belum terlalu dipakai)
* [helper.py](./helper.py) - Fungsi helper lain-lain