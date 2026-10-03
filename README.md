# Bitki Hastalığı Tespiti

[![CI](https://github.com/yusufsmnc/bitki-hastalik-tespiti/actions/workflows/ci.yml/badge.svg)](https://github.com/yusufsmnc/bitki-hastalik-tespiti/actions/workflows/ci.yml)

Yaprak fotoğrafından domates, patates ve biber hastalıklarını tanımaya çalışan,
**sınırlarını bilen** bir karar-destek sistemi.

Hedef kullanım: tarladaki bir çiftçi telefonuyla yaprağın fotoğrafını çeker,
sistem olası hastalığı ve ne kadar emin olduğunu söyler. Emin değilse bunu
açıkça söyler ve daha iyi bir fotoğraf ister.

---

## Önce dürüst tablo

Bu bölüm başta duruyor, çünkü projeyi değerlendirirken ilk bilmeniz gereken şey
bu. Bitki hastalığı tespiti literatüründe "%99 doğruluk" iddiaları yaygındır ve
neredeyse hepsi **laboratuvar** rakamıdır. Bu projede ikisini de ölçtük:

| Ortam | Doğruluk |
|---|---|
| Laboratuvar fotoğrafları (düz zemin, tek yaprak, kontrollü ışık) | ~%99 |
| **Gerçek tarla fotoğrafları** (dağınık arka plan, gölge, açı) | **~%62** |

Aradaki bu uçurum bir hata değil, bu alanın bilinen ve zor problemi:
laboratuvarda öğrenilen şey tarlada aynı işe yaramıyor. **Bu projenin asıl
rakamı %62'dir.** Herhangi bir yerde "%99 doğrulukla hastalık tespiti" ifadesi
görürseniz, o ifade yanlıştır.

### Peki %62 ile ne yapılır?

Modeli olduğu gibi kullanmak, çiftçiye vakaların üçte birinden fazlasında
yanlış bilgi vermek demekti. Bunun yerine sisteme **susmayı** öğrettik: model
yeterince emin değilse tahmin yapmıyor, "bu yaprağı net tanıyamadım" diyor.

Eşik keyfi seçilmedi, ölçüldü. 569 gerçek tarla fotoğrafı (PlantDoc test
bölümü) üzerinde:

| Güven eşiği | Kapsama (cevap verdiği oran) | Cevap verdiğinde isabet |
|---|---|---|
| 0.80 | %58.3 | %73.8 |
| 0.90 | %45.0 | %80.1 |
| **0.95 (seçilen)** | **%36.6** | **%84.6** |
| 0.97 | %30.2 | %87.8 |

Seçim kuralı: *isabetin ~%85'e ulaştığı en düşük eşik.* 0.97 daha isabetli ama
6.4 puan daha kapsama yakıyor; bu kazanç o bedeli karşılamıyor.

Bedeli açıkça kabul ediyoruz: **fotoğrafların yaklaşık %63'ünde sistem cevap
vermiyor.** Bu bir kusur değil, bilinçli bir takas — yanlış yönlendirmektense
susmak.

> **Bu rakamlar iyimser.** Eşik, yukarıdaki ölçümlerin yapıldığı *aynı* test
> setine bakılarak seçildi. Yani %84.6 isabet, görülmemiş yeni fotoğraflarda
> muhtemelen biraz daha düşük çıkar. Dürüst bir ölçüm için eşiğin ayrı bir
> doğrulama setinde seçilip test setinde bir kez ölçülmesi gerekir. Bunu
> [Yol haritası](#yol-haritası) bölümünde ilk sıraya koyduk.

### Bu sistem ne DEĞİL

- **Teşhis aracı değil.** Kesin tanı için ziraat mühendisine danışılmalı.
- **Genel amaçlı bitki tanıyıcı değil.** Sadece 3 ürün ve 15 sınıf biliyor.
  Bir fasulye yaprağı gösterirseniz model bunu "bilmiyorum" diye reddedemez;
  bildiği 15 sınıftan birine benzetmeye çalışır. Tek koruma güven eşiğidir.
- **Tarla koşullarında güvenilir değil.** Yukarıdaki tabloya bakın.

---

## Neyi başardık

Projenin bu noktaya kadar çözdüğü somut problemler:

- **Laboratuvar–tarla uçurumunu görünür kıldık.** Üç veri setini (biri
  laboratuvar, ikisi gerçek tarla) birleştirip modeli ikisinde de ölçtük.
  Çoğu örnek projenin atladığı adım bu.
- **Güven eşiğini veriyle seçtik.** `tune_threshold.py` ile kapsama/isabet
  dengesini ölçüp eşiği gerekçeli biçimde belirledik; tahminî bir sayı değil.
- **Sistemi "bilmiyorum" diyebilir hale getirdik.** Mimarinin en değerli
  parçası bu.
- **Uçtan uca çalışan bir zincir kurduk:** model → Python API → telefon
  uyumlu web arayüzü. Fotoğraf çekmekten sonucu görmeye kadar her adım çalışıyor.
- **Model dosyası olmadan çalışan test altyapısı yazdık.** Model git'te yok
  (45 MB), ama CI yine de her push'ta 41 testi çalıştırıyor.
- **Sonuçları abartmadan raporladık.** Yukarıdaki tablo bunun kanıtı.

---

## Veri setleri

Model üç ayrı veri setinin birleşimiyle eğitildi. Bu birleştirme projenin
merkezindeki fikir: tek başına laboratuvar verisi tarlada işe yaramıyor.

| Veri seti | Tür | Rolü |
|---|---|---|
| **PlantVillage** | Laboratuvar | Temiz, bol örnekli taban. Düz zeminde tek yaprak. |
| **PlantDoc** | Gerçek tarla | Dağınık arka plan, doğal ışık. Eşik ölçümü de bu setle yapıldı. |
| **PlantWild** | Gerçek tarla | Ek tarla çeşitliliği. |

### ⚠️ Lisans uyarısı

**PlantWild veri seti CC-BY-NC-ND lisanslıdır: ticari kullanım yasaktır.**

Bu model PlantWild verisiyle eğitildiği için, eğitilmiş ağırlıklar da bu
kısıtın etkisi altındadır. Projeyi ticari bir ürüne dönüştürmeyi
düşünüyorsanız modeli PlantWild olmadan yeniden eğitmeniz gerekir.

Diğer veri setlerinin lisans koşulları için kendi kaynaklarına bakın; bu
depoda veri seti dosyası bulunmuyor.

---

## Model

- **Mimari:** ResNet18 (torchvision)
- **Yöntem:** Transfer learning + fine-tuning
- **Çerçeve:** PyTorch
- **Çıktı katmanı:** 15 sınıf
- **Çıkarım (inference):** CPU. GPU gerekmez — tahmin saniyeler değil,
  milisaniyeler sürüyor.

> **Not:** Eğitim bu depoda yapılmıyor. Depo yalnızca *eğitilmiş modeli
> kullanan* servisi içerir; eğitim script'i burada yok. `best_model_v3.pth`
> (45 MB) ve `class_names.json` boyutları nedeniyle git'e dahil edilmedi
> (bkz. [Kurulum](#kurulum)).

### Ön işlemede kritik kural

Tahmin sırasındaki görüntü ön işlemesi, eğitimdekiyle **birebir** aynı olmak
zorundadır:

```python
transforms.Resize((224, 224))
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

Bu değerler farklı olursa model gözle görülür bir hata vermeden saçmalar —
en sık yapılan deploy hatasıdır. Bu yüzden değerleri bir test sabitliyor
(`tests/test_predict.py::test_on_isleme_egitimdekiyle_ayni_kalmali`).

---

## Tanınan sınıflar (15)

Model sınıf adlarını İngilizce (PlantVillage adlandırması) üretir; arayüzde
Türkçe karşılıkları gösterilir.

| Ürün | Sınıf | Türkçe |
|---|---|---|
| Biber | `Pepper__bell___Bacterial_spot` | Biber - Bakteriyel leke |
| Biber | `Pepper__bell___healthy` | Biber - Sağlıklı |
| Patates | `Potato___Early_blight` | Patates - Erken yanıklık |
| Patates | `Potato___Late_blight` | Patates - Geç yanıklık (mildiyö) |
| Patates | `Potato___healthy` | Patates - Sağlıklı |
| Domates | `Tomato_Bacterial_spot` | Domates - Bakteriyel leke |
| Domates | `Tomato_Early_blight` | Domates - Erken yanıklık |
| Domates | `Tomato_Late_blight` | Domates - Geç yanıklık (mildiyö) |
| Domates | `Tomato_Leaf_Mold` | Domates - Yaprak küfü |
| Domates | `Tomato_Septoria_leaf_spot` | Domates - Septoria yaprak lekesi |
| Domates | `Tomato_Spider_mites_Two_spotted_spider_mite` | Domates - Kırmızı örümcek |
| Domates | `Tomato__Target_Spot` | Domates - Hedef leke |
| Domates | `Tomato__Tomato_YellowLeaf__Curl_Virus` | Domates - Sarı yaprak kıvırcıklık virüsü |
| Domates | `Tomato__Tomato_mosaic_virus` | Domates - Mozaik virüsü |
| Domates | `Tomato_healthy` | Domates - Sağlıklı |

Listede olmayan bir sınıf adı gelirse (örn. model değişirse) uygulama
çökmez, ham İngilizce adı gösterir.

---

## Kurulum

### Gereksinimler

- Python 3.14 (CI bu sürümle çalışıyor; 3.11+ muhtemelen sorunsuzdur)
- Model dosyaları: `best_model_v3.pth` ve `class_names.json`

### 1. Model dosyalarını edinin

**Bu iki dosya git deposunda yoktur** — `best_model_v3.pth` 45 MB olduğu için
`.gitignore`'dadır. Dosyaları proje sahibinden edinip `bitki-backend/`
klasörünün içine koyun:

```
bitki-backend/
├── best_model_v3.pth
└── class_names.json
```

### 2. Sanal ortam ve bağımlılıklar

```bash
cd bitki-backend
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

pip install -r ../requirements.txt
```

CPU'da çalıştırmak yeterli olduğu için PyTorch'un CPU sürümü de kullanılabilir;
bu, ~2 GB'lık CUDA paketlerini indirmekten kurtarır:

```bash
pip install torch==2.14.0 torchvision==0.29.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

### 3. Sunucuyu başlatın

```bash
# bitki-backend/ klasöründeyken:
uvicorn main:app --reload
```

`main:app` komutu `main.py`'yi *import ederek* bulur, bu yüzden komutun
`bitki-backend/` içinden çalıştırılması gerekir. `--reload` yalnızca
geliştirme içindir.

Arayüz: **http://127.0.0.1:8000**

### Telefondan erişim

Aynı Wi-Fi ağındaki telefondan test etmek için sunucuyu tüm arayüzlere açın:

```bash
uvicorn main:app --host 0.0.0.0
```

Bilgisayarın yerel IP'sini öğrenip (`ipconfig` / `ifconfig`) telefonda
`http://192.168.x.x:8000` adresini açın. Güvenlik duvarı ilk seferde izin
isteyebilir.

---

## API

| Adres | Yöntem | Açıklama |
|---|---|---|
| `/` | GET | Telefon uyumlu web arayüzü |
| `/health` | GET | Sunucu ayakta mı — `{"durum": "calisiyor"}` |
| `/predict` | POST | Fotoğraf yükle, tahmin al |
| `/docs` | GET | FastAPI'nin otomatik ürettiği API arayüzü |
| `/static/...` | GET | Arayüz dosyaları |

### `/predict` kullanımı

```bash
curl -X POST -F "file=@yaprak.jpg" http://127.0.0.1:8000/predict
```

Model yeterince eminse (`guven` ≥ %95):

```json
{
  "durum": "basarili",
  "hastalik": "Tomato_Leaf_Mold",
  "hastalik_tr": "Domates - Yaprak küfü",
  "guven": 97.3
}
```

Emin değilse:

```json
{
  "durum": "emin_degil",
  "mesaj": "Bu yaprağı net tanıyamadım. Daha yakın ve net bir fotoğraf çeker misiniz?",
  "en_yakin_tahmin": "Tomato_Leaf_Mold",
  "en_yakin_tahmin_tr": "Domates - Yaprak küfü",
  "guven": 92.2
}
```

`en_yakin_tahmin` alanları bilgi amaçlıdır ve **arayüzde bilerek
gösterilmez**: sistem "tanıyamadım" dedikten sonra bir tahmin fısıldarsa
kullanıcı onu cevap sanar.

### Hata kodları

| Kod | Anlamı |
|---|---|
| 400 | Dosya okunabilir bir resim değil (bozuk, yarım veya resim olmayan) |
| 413 | Dosya 10 MB sınırını aşıyor |
| 422 | `file` alanı hiç gönderilmemiş |

Hatalar Türkçe bir `detail` alanıyla döner; arayüz bu metni doğrudan
kullanıcıya gösterir.
