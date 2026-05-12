# FruitVision

268 sınıflı meyve ve sebze veri seti üzerinde **ResNet-18** ve **Vision Transformer (ViT-B/16)** modellerinin eğitimi, değerlendirmesi ve Streamlit tabanlı karşılaştırma arayüzü.

## Model Sonuçları

| Model | Çözünürlük | Accuracy | F1-Score | Ortalama Gecikme |
| :--- | :--- | :--- | :--- | :--- |
| ResNet-18 | 224x224 | %97.77 | 0.9769 | 2.27 ms |
| ViT-B/16 | 128x128 | %99.95 | 0.9995 | 4.52 ms |

## Proje Yapısı

```text
├── app.py                        # Streamlit web arayüzü (Grad-CAM & Saliency Map)
├── data_unifier_v2.py            # Veri setlerini tek dizinde birleştirme betiği
├── unified_fruit_train_final.py  # ResNet-18 eğitim betiği
├── train_vit.py                  # ViT-B/16 eğitim betiği
├── evaluate_model.py             # ResNet-18 test ve hata matrisi betiği
├── evaluate_vit.py               # ResNet-18 ve ViT karşılaştırmalı değerlendirme betiği
├── live_predict.py               # Tekil görsel tahmini (CLI)
├── batch_predict.py              # Klasör bazlı toplu tahmin
├── models/                       # Eğitilmiş model ağırlıkları (.pth)
├── results/                      # Karşılaştırma grafikleri ve metrik çıktıları
└── requirements.txt              # Kütüphane bağımlılıkları
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

Veri setlerini birleştirmek için:

```bash
python3 data_unifier_v2.py
```

Modelleri eğitmek için:

```bash
python3 unified_fruit_train_final.py
python3 train_vit.py
```

Performans karşılaştırması ve karmaşıklık matrislerini üretmek için:

```bash
python3 evaluate_vit.py
```

Web arayüzünü başlatmak için:

```bash
streamlit run app.py
```

Komut satırından tek bir görseli test etmek için:

```bash
python3 live_predict.py <gorsel_yolu>
```

