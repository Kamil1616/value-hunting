# Value Hunting — İY/MS Analiz Sistemi

## Kurulum (Render.com)

### 1. GitHub'a yükle
```bash
git init
git add .
git commit -m "Value Hunting v2.0"
git remote add origin https://github.com/KULLANICI_ADIN/value-hunting.git
git push -u origin main
```

### 2. Render.com'da deploy
1. render.com'a git → New Web Service
2. GitHub repo'nu bağla
3. Environment Variables ekle:
   - Key: `ANTHROPIC_API_KEY`
   - Value: `sk-ant-...` (Anthropic API key'in)
4. Deploy!

## Kullanım

### Görsel Modu Seçimi:
- **Çift Mod** (Önerilen): Her maç için 2 görsel
  - 1. görsel: Genel sekmesi
  - 2. görsel: İç & Dış sekmesi
  
- **Tekli Mod**: Sadece 1 sekme varsa

### 50 Maç Yükleme:
- 100 görsel (50 maç × 2) seç
- Çift Mod aktif olmalı
- Analiz Et'e bas

## Model Parametreleri (Sabit)
- W_SPEC = 0.55 (iç/dış saha ağırlığı)
- W_GEN = 0.45 (genel ağırlık)  
- HT_FACTOR = 0.48 (İY lambda)
- GS_LEADER = 1.08 (lider koruma)
- PAYOUT = 0.90 (%90 payout)
- DC_RHO = -0.13 (Dixon-Coles)
