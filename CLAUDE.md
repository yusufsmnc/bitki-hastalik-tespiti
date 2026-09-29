# CLAUDE.md — Bitki Hastalığı Tespiti Projesi

Bu dosya, bu projede çalışan Claude (agent) için bağlam ve kurallardır. Her oturumda önce bunu oku, sonra çalış. Kullanıcı (proje sahibi) yeni başlayan bir geliştirici ve **öğrenmek** istiyor — senin işin kod üretmek değil, kullanıcının anlayarak ilerlemesine yardım etmek. Ürettiğin her şeyi kullanıcı okuyup onaylayacak.

---

## 1. Proje nedir

Yaprak fotoğrafından bitki hastalığı tespiti yapan bir sistem. Hedef kullanıcı: tarladaki bir çiftçi telefonundan yaprak fotoğrafı çeker, sistem hastalığı söyler.

- **Kapsam:** 3 ürün, 15 sınıf (domates, patates, biber — sağlıklı + çeşitli hastalıklar).
- **Model:** ResNet18, transfer learning + fine-tuning ile eğitildi (PyTorch).
- **Eğitim verisi:** PlantVillage (laboratuvar) + PlantDoc (gerçek tarla) + PlantWild (gerçek tarla) birleştirilerek.
- **Model dosyası:** `best_model_v3.pth` + `class_names.json`. **Bu dosyalar git'te YOK** (`.gitignore`'da, boyut nedeniyle). Google Drive'da tutuluyor; kullanıcı yerelde `bitki-backend/` içine koydu.

### Dürüst performans tablosu (ASLA abartma)
- Laboratuvar test doğruluğu: ~%99
- **Gerçek tarla test doğruluğu: ~%62** — asıl rakam bu.
- Bu bir "tanı makinesi" değil, **sınırlarını bilen bir karar-destek asistanı**. Güven eşiği altında "bu yaprağı net tanıyamadım, daha iyi fotoğraf çek" der.
- Herhangi bir metinde (README, arayüz, yorum) "%99 doğrulukla hastalık tespiti" gibi abartılı iddia YAZMA. Gerçek sınırları (laboratuvar-tarla uçurumu, %62, sadece 3 ürün) dürüstçe belirt.

---

## 2. Şu anki durum (tamamlanan)

- `bitki-backend/predict.py` — model yükleme (script başında, bir kere) + `tahmin_et_goruntu(img)` fonksiyonu. Görüntü nesnesi alır, softmax ile güven skoru hesaplar, güven eşiği (`GUVEN_ESIGI` = 0.95, Faz 2'de veriyle seçildi) altında "emin_degil" döndürür.
- `bitki-backend/main.py` — FastAPI. `/health` ve `/predict` (dosya yükleyip tahmin) endpoint'leri çalışıyor. `/predict` EXIF yönünü düzeltir, bozuk dosyaya 400, 10 MB üstüne 413 döner.
- Git deposu kurulu, GitHub'a bağlı: `yusufsmnc/bitki-hastalik-tespiti`.
- `requirements.txt` + `requirements-dev.txt` (depo kökünde), `.gitignore`, testler (`tests/`) ve CI (`.github/workflows/ci.yml`) hazır.

### Klasör yapısı
```
bitki-hastalik-tespiti/         (git deposu kökü)
├── .gitignore
├── README.md
├── requirements.txt            (uygulama bağımlılıkları)
├── requirements-dev.txt        (pytest, httpx)
├── pytest.ini
├── .github/workflows/ci.yml
├── tests/                      (conftest.py sahte model üretir)
├── bitki-backend/
│   ├── venv/                   (git yok)
│   ├── predict.py
│   ├── main.py
│   ├── tune_threshold.py
│   ├── plantdoc_split/         (git yok — eşik ayarı için test görüntüleri)
│   ├── best_model_v3.pth       (git yok — yerelde var)
│   ├── class_names.json        (git yok — yerelde var)
│   └── test.jpg                (git yok)
```

---

## 3. Teknoloji ve gerekçeleri

- **Python + FastAPI:** Model PyTorch'ta (Python), o yüzden servis de Python olmalı ki model ile arasında çeviri gerekmesin. FastAPI modern, hızlı, otomatik `/docs` üretir, ML'de fiili standart.
- **PyTorch (CPU inference):** Backend model eğitmez, sadece yükleyip tahmin yapar. Tahmin CPU'da hızlı; GPU gerekmez. Modeli `map_location="cpu"` ile yükle.
- **Ön işleme birebir aynı olmalı:** Tahmindeki resize (224x224) ve normalize değerleri, eğitimdeki `eval_transform` ile BİREBİR aynı olmalı (`Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])`). Farklı olursa model saçmalar — bu en sık deploy hatasıdır.

---

## 4. Kritik kısıtlar (agent bunları BİLMEDEN iş yaparsa hata çıkar)

1. **CI'da model dosyası yok.** `best_model_v3.pth` git'te olmadığı için, GitHub Actions ortamında yüklenemez. Testleri buna göre kur: modeli mock'la, ya da küçük sahte bir model üret, ya da testleri model gerektirmeyen kısımlara odakla. Gerçek `.pth`'e bağlı test CI'da patlar.
2. **`.gitignore`'a dokunma dikkatli.** `*.pth`, `venv/`, `test.jpg`, `__pycache__/` git'e ASLA girmemeli. Yeni büyük/gizli dosya eklersen `.gitignore`'a da ekle.
3. **CORS production'da kısıtlanmalı.** Geliştirmede tüm origin'lere izin verilebilir ama bunu her zaman yorumla işaretle: "production'da kısıtla".
4. **Model her istekte değil, uygulama başında bir kere yüklenir.** `tahmin_et_goruntu` içinde model yükleme YAPMA.

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
