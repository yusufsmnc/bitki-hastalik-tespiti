# CLAUDE.md — Bitki Hastalığı Tespiti Projesi

Bu dosya, bu projede çalışan Claude (agent) için bağlam ve kurallardır. Her oturumda önce bunu oku, sonra çalış. Kullanıcı (proje sahibi) yeni başlayan bir geliştirici ve **öğrenmek** istiyor — senin işin kod üretmek değil, kullanıcının anlayarak ilerlemesine yardım etmek. Ürettiğin her şeyi kullanıcı okuyup onaylayacak.

---

## 1. Proje nedir

Yaprak fotoğrafından bitki hastalığı tespiti yapan bir sistem. Hedef kullanıcı: tarladaki bir çiftçi telefonundan yaprak fotoğrafı çeker, sistem hastalığı söyler.

- **Kapsam:** 3 ürün, 15 sınıf (domates, patates, biber — sağlıklı + çeşitli hastalıklar).
- **Model:** ResNet18, transfer learning + fine-tuning ile eğitildi (PyTorch).
- **Eğitim verisi:** PlantVillage (laboratuvar) + PlantDoc (gerçek tarla) + PlantWild (gerçek tarla) birleştirilerek.
- **Model dosyası:** `best_model_v3_durust.pth` **git'te YOK** (`.gitignore`'da, boyut nedeniyle). Google Drive'da tutuluyor; kullanıcı yerelde `bitki-backend/` içine koydu. Eski `best_model_v3.pth` da yerelde duruyor (geri dönüş kopyası), artık kullanılmıyor.
- **`class_names.json` git'te VAR** (407 bayt, sınıf adları). Sınıf sırası modelle birebir eşleşmeli: yeni model gelirse bu dosya da birlikte güncellenir.
- **`model_config.json` git'te VAR** (model dosya adı, sıcaklık, iki karar eşiği `guven_esigi`/`saglikli_esigi`, bir de ürün tutarlılık eşiği `urun_esigi`). Bu sayılar kodun değil **modelin** özelliği — hepsi bu belirli ağırlık dosyası üzerinde ölçüldü. Yeni model gelirse üçü birlikte yeniden ölçülür.

### Karar kuralı: ürün maskesi + kalibrasyon

1. **Ürün maskesi.** Kullanıcı ürünü seçer (`domates` / `patates` / `biber`); seçilen ürünün dışındaki sınıfların logitleri `-inf` yapılır. Maske **softmax'tan ÖNCE** uygulanır — sonra uygulanırsa kesilen sınıfların olasılığı pay toplamına girer, kalan olasılıklar 1'e toplanmaz ve güven skorları ölçülen değerlerle uyumsuz olur.
2. **Sıcaklık ölçekleme.** Logitler softmax'tan önce `T = 1.95`'e bölünür. Tahmini **değiştirmez** (bölme sıralamayı bozmaz), sadece aşırı özgüvenli olasılıkları bastırıp güven skorunu gerçek isabetle uyumlu hale getirir.
3. **İki eşik.** En yüksek olasılık `>= 0.70` ise kesin cevap. Tahmin bir `healthy` sınıfsa daha sıkı eşik: `0.80`. Altında "emin değilim" denir ve en olası 3 aday gösterilir.
4. **Ürün tutarlılık uyarısı.** Maskeden **ÖNCE** (tüm sınıflar açık), `T`-ölçekli softmax'ta seçilen ürünün sınıflarına düşen olasılık toplamı ("ürün payı") hesaplanır. Pay `urun_esigi = 0.20`'nin altındaysa muhtemelen yanlış ürün seçilmiştir (model olasılığı başka bir ürüne yığmış); cevaba `urun_uyarisi: true` eklenir ve arayüz "bu fotoğraf seçtiğiniz bitkiye benzemiyor" uyarısı gösterir. **Uyarı sonucu ENGELLEMEZ, durumu/tahmini/adayları DEĞİŞTİRMEZ** — yalnızca ek bir bayrak; tahmin yolu maskeli softmax'tan bağımsız işler. Pay maskeden sonra hesaplanamaz (maske diğerlerini sıfırlar, toplam her zaman 1 olurdu); `T`'ye bölme ölçümdeki gibi burada da uygulanır.
5. Değerler `model_config.json`'da. **Eşik eskiden 0.95'ti, şimdi 0.70 — bu bir GEVŞETME DEĞİL:** 0.95 kalibre edilmemiş (aşırı özgüvenli) olasılıklar üzerindeydi, 0.70 ise `T` ile bastırılmış olasılıklar üzerinde. Ölçekler farklı, sayıları doğrudan karşılaştırma.

### Dürüst performans tablosu (ASLA abartma)

`best_model_v3_durust.pth` ölçümleri (aksi yazılmayan satırlar: PlantDoc **saha** testi, 569 görüntü):

| Ölçüm | Değer |
|---|---|
| Laboratuvar (PlantVillage val, 3.089, seçimde kullanılmadı) | %97.64 |
| Top-1, ürün maskesi **olmadan** | %57.64 |
| Top-1, ürün maskesi **ile** | %69.77 |
| Kesin cevapların (`basarili`) isabeti | %88.1 |
| "Sağlıklı" kesin cevaplarının isabeti | %83.3 |
| Eşik altında (`emin_degil`) doğru cevabın adaylar içinde olma oranı | %88.4 |

**Laboratuvar ile maskesiz saha arasındaki fark (%97.6 → %57.6) yeni modelde de sürüyor.** "Dürüst" model uçurumu kapatmadı; onu görünür kıldı ve ürün maskesi + kalibrasyonla yönetilebilir hale getirdi.

**Bu sayıları yazarken bağlamını da ver — tek başına aktarılırsa yanıltıcı olur:**

- `T = 1.95` ve `0.70` eşiği, PlantDoc **eğitim** bölümünden ayrılmış **163 görüntülük ayrı saha val setinde** seçildi; test setiyle phash karşılaştırılıp kopyaları temizlendi. 569 görüntülük test seti seçimde kullanılmadı, sadece raporlama için bir kez kullanıldı.
- `0.80` "sağlıklı" eşiği **veriyle ayarlanmadı** — bilinçli bir güvenlik kararı: hastalıklı yaprağa "sağlıklı" demek tedaviyi geciktirir, bu hatayı zorlaştırmak istedik. Val'da "sağlıklı" tahmini sayısı 5–17 arasındaydı; ayrı bir eşik ayarlamaya yetmez.
- **Kalan sınırlar:** val seti küçük (163) olduğu için eşik seçimi gürültülü; modelin epoch'u (8) da aynı saha val setinde seçildi — yani laboratuvar val'i (PlantVillage, 3.089) hiçbir seçimde kullanılmadı, bir doğrulama bölümüdür ama test bölümü değildir; test sonuçları tek bir 569 görüntülük setten geliyor; test etiketlerinin bir kısmının gürültülü olduğu bulundu.
- **Varsayım:** çiftçinin ürünü **doğru seçtiği** varsayılıyor. Yanlış ürün seçilirse maske doğru sınıfı keser ve yukarıdaki sayılar geçersizdir. Bu riske karşı **ürün tutarlılık uyarısı** var (karar kuralı adım 4): `urun_esigi = 0.20` saha val'da seçildi (yakalama %85.6, yanlış alarm %4.3); test (569) yakalama %82.8, yanlış alarm %4.6; yanlış ürünle emin görünen yanlış cevap oranı **%55.3 → %7.5**. **Genel %4.6'yı tek başına yazma — ürüne göre dağılımı da ver:** domates (10 sınıf) %1.6, patates (3 sınıf) %7.2, biber (2 sınıf) %17.6 (test, doğru ürün seçildiğinde; biber n=51, kesin değil). Pay ürünün sınıf sayısından etkilenir; 2 sınıflı biber tek eşikte dezavantajlı, bu yüzden biberde uyarı daha sık çıkar. Şimdilik tek eşik: saha val'da yalnızca 15 biber görüntüsü var, ürüne özel eşik ayarlamaya yetmez; uyarı engellemediği (yalnızca sorduğu) için fazla alarm güvenliği bozmaz. Sınıf sayısına göre düzeltme yol haritasında (Aşama 2).

Diğer dürüstlük kuralları:

- Bu bir "tanı makinesi" değil, **sınırlarını bilen bir karar-destek asistanı**. Güven eşiği altında "bu yaprağı net tanıyamadım, daha iyi fotoğraf çek" der ve adayları listeler.
- Herhangi bir metinde (README, arayüz, yorum) "%99 doğrulukla hastalık tespiti" gibi abartılı iddia YAZMA. Gerçek sınırları (sadece 3 ürün, saha doğruluğu, ürün seçimi varsayımı) dürüstçe belirt.
- **Laboratuvar-tarla uçurumu:** önceki modelde (`best_model_v3`) laboratuvar ~%99, gerçek tarla ~%62 ölçülmüştü — uçurumun somut kanıtı bu. Yeni model için laboratuvar rakamı ölçülmedi; **uydurma**, yukarıdaki tabloda olmayan bir sayı yazma.

---

## 2. Şu anki durum (tamamlanan)

- `bitki-backend/predict.py` — model ve ayar yükleme (script başında, bir kere) + `tahmin_et_goruntu(img, urun)` fonksiyonu. Görüntü nesnesi ve ürün adı alır; ürün maskesi → `T`'ye bölme → softmax sırasıyla olasılık hesaplar, eşiğin altında "emin_degil" döndürür. Ayrıca maskeden **önce** ürün payını ölçüp pay `URUN_ESIGI`'nin altındaysa cevaba `urun_uyarisi: true` ekler (karar kuralı adım 4); bu bayrak durumu/tahmini değiştirmez. Geçersiz ürün için `ValueError` fırlatır. Ayarlar `model_config.json`'dan okunur, modül sabitlerine (`SICAKLIK`, `GUVEN_ESIGI`, `SAGLIKLI_ESIGI`, `URUN_ESIGI`) atanır. Ürün → sınıf eşlemesi `class_names.json`'daki adların önekinden türetilir (`Tomato`/`Potato`/`Pepper`), elle yazılmaz.
- `bitki-backend/main.py` — FastAPI. `/health` ve `/predict` endpoint'leri çalışıyor. `/predict` EXIF yönünü düzeltir, bozuk dosyaya 400, 10 MB üstüne 413 döner.
- `bitki-backend/static/index.html` — telefon arayüzü (Faz 3). Akışın ilk adımı **ürün seçimi**: üç düğme (Domates/Patates/Biber), biri seçilene kadar fotoğraf düğmesi pasif ve nedeni metinle yazılı. İstekte `urun` alanı gönderilir. Her sonuç kartında **"Seçilen bitki: …"** satırı görünür (`veri.urun`) — yanlış ürün seçimi en olası kullanıcı hatası ve çiftçinin bunu fark etmesinin yolu bu satır. `emin_degil` durumunda **aday listesi** gösterilir (Türkçe adlar + tam sayı yüzdeler), `en_yakin_tahmin` gösterilmez. **`urun_uyarisi: true` gelirse** sonuç kartının üstünde kehribar bir uyarı kutusu açılır ("bu fotoğraf seçtiğiniz bitkiye (…) pek benzemiyor") + "Bitkiyi değiştir" düğmesi; kutu `DURUM_BOLUMLERI`'ne dahil olduğu için her ekran geçişinde (ürün değişince de) otomatik gizlenir, mesaj `textContent` ile yazılır. Sonucu engellemez, yalnızca sorar.
- **Ürün değişirse istek KENDİLİĞİNDEN gönderilmez.** Eski sonuç kartı gizlenir (artık seçili bitkiyle çelişir) ve fotoğraf hâlâ eldeyse bir onay kartı çıkar: "Bu fotoğrafı <Bitki> olarak gönder" / "Yeni fotoğraf çek". Gerekçe: "yanlış bitki seçtim, düzeltiyorum" ile "bu bitkiyi bitirdim, diğerine geçiyorum" niyeti dışarıdan aynı görünür (ikisinde de bir ürün düğmesine basılır); otomatik gönderim ikinci durumda eski fotoğrafı yeni bitki olarak yollar ve tam da kaçınmak istediğimiz hatayı üretir. Sonuç kartlarındaki "Bitkiyi değiştir" düğmesi yalnızca ürün seçicisine götürür, istek göndermez. `tests/test_sozlesme.py` bu YOKLUĞU da test ediyor.
- Git deposu kurulu, GitHub'a bağlı: `yusufsmnc/bitki-hastalik-tespiti`.
- `requirements.txt` + `requirements-dev.txt` (depo kökünde), `.gitignore`, testler (`tests/`) ve CI (`.github/workflows/ci.yml`) hazır.

#### `/predict` sözleşmesi
- **İstek:** `file` (resim) + **`urun`** (`domates` / `patates` / `biber`) — ikisi de **ZORUNLU** form alanı. Eksik alan → 422 (FastAPI'nin kendi doğrulaması). Geçersiz ürün değeri → **400** + okunur Türkçe `detail` metni (422'nin `detail`'i liste döner, arayüz metin bekliyor).
- **Cevap (`durum: "basarili"`):** `durum`, `urun`, `urun_uyarisi`, `hastalik`, `hastalik_tr`, `guven`, `adaylar`
- **Cevap (`durum: "emin_degil"`):** `durum`, `urun`, `urun_uyarisi`, `mesaj`, `en_yakin_tahmin`, `en_yakin_tahmin_tr`, `guven`, `adaylar`
- **`urun_uyarisi`:** her iki durumda da bulunan bool. `true` ise maskeden önceki ürün payı `urun_esigi`'nin altında kalmış, yani muhtemelen yanlış ürün seçilmiş. Payın kendisi döndürülmez.
- **`adaylar`:** en olası en fazla 3 sınıf, her biri `{"hastalik", "hastalik_tr", "guven"}`. `guven` birimi diğer `guven` alanıyla aynı: yüzde, 1 ondalık. Maskelenmiş sınıflar listeye girmez — biberin 2 sınıfı olduğu için biberde 2 aday döner.
- Arayüzde `en_yakin_tahmin` **gösterilmez**, `adaylar` gösterilir: tek bir tahmin cevap gibi görünür, liste ise belirsizliği açıkça gösterir (eşik altında doğru cevap %88.4 oranında adayların içinde). Bu bir ürün kararı, `tests/test_sozlesme.py` ile sabitlenmiş.

### Klasör yapısı
```
bitki-hastalik-tespiti/         (git deposu kökü)
├── .gitignore
├── README.md
├── requirements.txt            (uygulama bağımlılıkları)
├── requirements-dev.txt        (pytest, httpx)
├── pytest.ini
├── .github/workflows/ci.yml
├── docs/                       (yol haritası, tarihsel kayıtlar)
├── tests/                      (conftest.py sahte model üretir)
├── bitki-backend/
│   ├── venv/                   (git yok)
│   ├── predict.py
│   ├── main.py
│   ├── tune_threshold.py       (ESKİ kurala göre ölçer — maske ve T yok)
│   ├── static/index.html       (telefon arayüzü, tek dosya)
│   ├── plantdoc_split/         (git yok — eşik ayarı için test görüntüleri)
│   ├── best_model_v3_durust.pth (git yok — kullanılan model)
│   ├── best_model_v3.pth       (git yok — eski, geri dönüş kopyası)
│   ├── model_config.json       (git'te var — sıcaklık ve eşikler)
│   ├── class_names.json        (git'te var — modelle eşleşmeli)
│   └── test.jpg                (git yok)
```

---

## 3. Teknoloji ve gerekçeleri

- **Python + FastAPI:** Model PyTorch'ta (Python), o yüzden servis de Python olmalı ki model ile arasında çeviri gerekmesin. FastAPI modern, hızlı, otomatik `/docs` üretir, ML'de fiili standart.
- **PyTorch (CPU inference):** Backend model eğitmez, sadece yükleyip tahmin yapar. Tahmin CPU'da hızlı; GPU gerekmez. Modeli `map_location="cpu"` ile yükle.
- **Ön işleme birebir aynı olmalı:** Tahmindeki resize (224x224) ve normalize değerleri, eğitimdeki `eval_transform` ile BİREBİR aynı olmalı (`Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])`). Farklı olursa model saçmalar — bu en sık deploy hatasıdır.

---

## 4. Kritik kısıtlar (agent bunları BİLMEDEN iş yaparsa hata çıkar)

1. **CI'da model dosyası yok.** `best_model_v3_durust.pth` git'te olmadığı için, GitHub Actions ortamında yüklenemez. Testleri buna göre kur: modeli mock'la, ya da küçük sahte bir model üret, ya da testleri model gerektirmeyen kısımlara odakla. Gerçek `.pth`'e bağlı test CI'da patlar.
2. **`.gitignore`'a dokunma dikkatli.** `*.pth`, `venv/`, `test.jpg`, `__pycache__/` git'e ASLA girmemeli. Yeni büyük/gizli dosya eklersen `.gitignore`'a da ekle.
3. **CORS production'da kısıtlanmalı.** Geliştirmede tüm origin'lere izin verilebilir ama bunu her zaman yorumla işaretle: "production'da kısıtla".
4. **Model her istekte değil, uygulama başında bir kere yüklenir.** `tahmin_et_goruntu` içinde model yükleme YAPMA.
5. **Maskesiz tahmin yolu açılmamalı.** Sıcaklık (`T = 1.95`) ve karar eşikleri (`0.70` / `0.80`) yalnızca **ürün maskesi uygulanmış** çıktılar üzerinde ölçüldü. `urun` alanını opsiyonel yapmak veya maskeyi atlayan bir yol eklemek, bu sayıları geçersiz kılar; sistem kalibre olduğunu sanarak kalibre olmayan güven skorları gösterir. (Ürün payı maskeden **önce** hesaplanır ama o da seçili `urun`'u gerektirir — maskesiz genel bir tahmin yolu yine yok.) **Model değişirse `T`, iki karar eşiği ve `urun_esigi` birlikte yeniden ölçülmeli** — eski değerleri yeni ağırlıklarla kullanma.

---

## 5. Git akışı kuralları (her iş için)

Her yeni iş, doğrudan `main`'e değil, kendi dalında yapılır:

1. `git checkout -b <tip>/<kısa-ad>` (örn. `feat/mobile-frontend`)
2. İşi yap, küçük ve anlamlı commit'ler at
3. `git push -u origin <dal-adı>`
4. GitHub'da PR aç (açıklama yaz)
5. CI'ın (varsa) yeşil olmasını bekle
6. Merge et
7. `git checkout main && git pull`, sonra dalı sil

**main'de doğrudan çalışma.** Her zaman önce dal aç.

---

## 6. Commit kuralları — İngilizce Conventional Commits

Format: `tip: emir kipinde kısa açıklama` (küçük harf, sonuna nokta yok, ~50 karakter)

Tipler:
- `feat:` yeni özellik — `feat: add /predict endpoint`
- `fix:` hata düzeltme — `fix: correct image normalization`
- `chore:` bakım/yapılandırma — `chore: add requirements.txt`
- `docs:` dokümantasyon — `docs: complete README`
- `test:` test ekleme/düzeltme — `test: add tests for tahmin_et_goruntu`
- `refactor:` davranış değişmeden yeniden düzenleme — `refactor: extract prediction into function`

Bir commit tek bir mantıksal değişiklik olmalı. "Her şeyi tek commit'e" tıkma.

---

## 7. Kod kuralları

- Türkçe değişken/fonksiyon isimleri mevcut kodda kullanılıyor (`tahmin_et_goruntu`, `guven`, `GUVEN_ESIGI`) — tutarlılık için bu tarzı koru.
- Kullanıcı yeni başlayan; ürettiğin koda kısa açıklayıcı yorumlar ekle ama boğma.
- Yeni kütüphane eklersen `requirements.txt`'i güncelle.
- Basit ve okunur tut; gereksiz soyutlama/karmaşıklık ekleme.

---

## 8. Fazlar (sırayla, her biri kendi dalı + PR'ı)

### Faz 0 — Git akışı provası
Dal: `chore/add-readme-skeleton`. Kısa bir README iskeleti oluştur. Amaç: kullanıcı dal→PR→merge akışını düşük riskle bir kez yaşasın. Commit: `docs: add README skeleton`.

### Faz 1 — Testler + CI
Dal: `test/add-ci-and-first-tests`.
- `tests/test_predict.py`: `tahmin_et_goruntu` dönen sözlükte `durum` ve `guven` var mı, `guven` 0-100 arası mı test et. **CI'da model yok** (bkz Kısıt 1) — mock veya sahte model kullan.
- `.github/workflows/ci.yml`: her push/PR'da Python kur, requirements yükle, `pytest` çalıştır.
- Commit'ler: `test: add first tests for tahmin_et_goruntu`, `chore: add GitHub Actions CI workflow`.

### Faz 2 — Güven eşiğini veriyle ayarla
Dal: `feat/tune-confidence-threshold`.
- `tune_threshold.py`: etiketli test görüntülerini modelden geçir, her biri için (gerçek etiket, tahmin, güven, doğru/yanlış) topla. Farklı eşikler (0.5–0.8) için "kaç tahmin emin sayılır, yüzde kaçı doğru" tablosu üret.
- **Bu script'i kullanıcı çalıştırır** (model onda). Kullanıcı tabloya bakıp eşiğe karar verir, `predict.py`'deki `GUVEN_ESIGI` güncellenir.
- Commit'ler: `feat: add threshold tuning script`, `fix: set confidence threshold based on data`.

### Faz 3 — Telefon arayüzü + CORS
Dal: `feat/mobile-frontend`.
- `main.py`'ye CORS middleware (geliştirme için açık, yorumla "production'da kısıtla").
- `static/index.html`: telefon uyumlu tek sayfa. Fotoğraf çek/yükle (`input type=file accept="image/*" capture`), `/predict`'e POST, dönen JSON'a göre sonucu göster (`basarili` → hastalık + güven %; `emin_degil` → uyarı). Basit, temiz, Türkçe.
- FastAPI `static/` klasörünü servis etsin.
- Commit'ler: `feat: add CORS middleware`, `feat: add mobile upload interface`, `feat: serve static files`.

### Faz 4 — README
Dal: `docs/complete-readme`.
- Proje ne yapıyor (1 cümle), veri setleri (PlantVillage + PlantDoc + PlantWild, **PlantWild lisansı CC-BY-NC-ND: ticari kullanım yasak** notu), model (ResNet18, transfer learning), sonuçlar (**dürüst: laboratuvar %99 → tarla %62, sadece 3 ürün, laboratuvar-tarla uçurumu**), nasıl çalıştırılır (venv, requirements, uvicorn), API kullanımı, teknolojiler.
- **Abartma yasak** (bkz Bölüm 1). Commit: `docs: complete project README`.

### Faz 5 — Deploy (opsiyonel)
Dal: `chore/deploy-setup`. Hugging Face Spaces / Render / Railway. Model dosyasını platforma ayrıca yükle (git'te yok). CORS'u production için kısıtla.

---

## 9. İnsanla çalışma kuralları

- **Küçük görevler.** Bir seferde bir net iş yap, sonra dur ve kullanıcının kontrol etmesini bekle. "Tüm backend'i bitir" gibi toplu iş yapma.
- **Açıkla.** Ne yaptığını ve neden yaptığını kısaca anlat. Kullanıcı "bu ne yapıyor?" diye sorabilir — sabırla açıkla.
- **Şüphede dur ve sor.** Belirsiz bir karar (dosya silme, mimari seçim, geri alınamaz işlem) varsa önce sor.
- **Sık commit öner.** Her anlamlı adımdan sonra commit'lemesini hatırlat.
- **Dürüstlük.** Bir şey çalışmıyorsa veya emin değilsen söyle. Uydurma. Test etmeden "çalışıyor" deme.
