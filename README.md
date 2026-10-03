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
