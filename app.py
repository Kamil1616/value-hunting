import os
import math
import base64
import json
import re
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__)

# Grok API — OpenAI formatında çalışır
client = OpenAI(
    api_key=os.environ.get("GROK_API_KEY"),
    base_url="https://api.x.ai/v1"
)

# ══════════════════════════════════════════════════════════════
# SABİT KALİBRE PARAMETRELERİ
# ══════════════════════════════════════════════════════════════
W_SPEC     = 0.55   # iç/dış saha ağırlığı
W_GEN      = 0.45   # genel ağırlık
HT_FACTOR  = 0.48   # ilk yarı lambda katsayısı
GS_LEADER  = 1.08   # lider takım skoru koruma faktörü
GS_DRAW    = 1.04   # beraberden çıkma faktörü
LEAGUE_AVG = 1.20   # lig ortalaması
PAYOUT     = 0.90   # %90 payout marjı
DC_RHO     = -0.13  # Dixon-Coles düzeltme katsayısı


# ══════════════════════════════════════════════════════════════
# VALUE HUNTING MODELİ
# ══════════════════════════════════════════════════════════════
def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def dixon_coles(i, j, lh, la, rho=DC_RHO):
    if i == 0 and j == 0: return 1 - lh * la * rho
    elif i == 1 and j == 0: return 1 + la * rho
    elif i == 0 and j == 1: return 1 + lh * rho
    elif i == 1 and j == 1: return 1 - rho
    return 1.0


def result(i, j):
    return '1' if i > j else ('X' if i == j else '2')


def avg(lst, idx):
    if not lst:
        return 1.0
    return sum(x[idx] for x in lst) / len(lst)


def value_hunting_model(home_genel, home_ic, away_genel, away_dis):
    """
    Ana Value Hunting modeli.
    Girdi: her takım için (attığı, yediği) tuple listesi
    Çıktı: İY/MS olasılıkları sıralı liste
    """
    # Ağırlıklı hücum/savunma ortalamaları
    h_att = W_SPEC * avg(home_ic, 0)  + W_GEN * avg(home_genel, 0)
    h_def = W_SPEC * avg(home_ic, 1)  + W_GEN * avg(home_genel, 1)
    a_att = W_SPEC * avg(away_dis, 0) + W_GEN * avg(away_genel, 0)
    a_def = W_SPEC * avg(away_dis, 1) + W_GEN * avg(away_genel, 1)

    # Geometrik ortalama ile lambda (daha stabil)
    lam_h = math.sqrt((h_att / LEAGUE_AVG) * (a_def / LEAGUE_AVG)) * LEAGUE_AVG
    lam_a = math.sqrt((a_att / LEAGUE_AVG) * (h_def / LEAGUE_AVG)) * LEAGUE_AVG

    # Poisson + Dixon-Coles skor matrisi
    score_probs = {}
    for i in range(7):
        for j in range(7):
            p = poisson_pmf(i, lam_h) * poisson_pmf(j, lam_a)
            p *= dixon_coles(i, j, lam_h, lam_a)
            score_probs[(i, j)] = p
    total = sum(score_probs.values())
    score_probs = {k: v / total for k, v in score_probs.items()}

    # İlk yarı Poisson matrisi
    ht_probs = {}
    for i in range(5):
        for j in range(5):
            ht_probs[(i, j)] = poisson_pmf(i, lam_h * HT_FACTOR) * poisson_pmf(j, lam_a * HT_FACTOR)
    ht_total = sum(ht_probs.values())
    ht_probs = {k: v / ht_total for k, v in ht_probs.items()}

    # İY/MS kombinasyonları + game state faktörü
    iyms = {}
    for (hi, hj), hp in ht_probs.items():
        for (fi, fj), fp in score_probs.items():
            key = f"{result(hi, hj)}/{result(fi, fj)}"
            ht_r = result(hi, hj)
            ft_r = result(fi, fj)
            if ht_r == ft_r and ht_r != 'X':
                gs = GS_LEADER
            elif ht_r == 'X' and ft_r != 'X':
                gs = GS_DRAW
            else:
                gs = 1.0
            iyms[key] = iyms.get(key, 0) + hp * fp * gs

    total_iyms = sum(iyms.values())
    iyms = {k: v / total_iyms for k, v in iyms.items()}

    sorted_iyms = sorted(iyms.items(), key=lambda x: x[1], reverse=True)

    return sorted_iyms, lam_h, lam_a


def format_bar(pct, max_len=20):
    """Olasılık barı oluştur"""
    filled = int(pct * max_len / 35)
    filled = min(filled, max_len)
    return '▓' * filled + '░' * (max_len - filled)


def build_output(home_name, away_name, sorted_iyms, lam_h, lam_a):
    """Standart çıktı formatı oluştur"""
    lines = []
    W = 52
    lines.append("━" * W)
    lines.append(f"  MAÇ · {home_name} vs {away_name}")
    lines.append(f"  λ_ev={lam_h:.3f}  |  λ_dep={lam_a:.3f}")
    lines.append("━" * W)
    lines.append(f"  {'#':<4} {'İY/MS':<6} {'OLASILIK':<24} {'ORAN':>6}")
    lines.append("━" * W)

    for idx, (combo, prob) in enumerate(sorted_iyms, 1):
        oran = (1 / prob) * PAYOUT
        pct = prob * 100
        bar = format_bar(pct)
        if idx == 6:
            lines.append(f"  · · · · · · · · · · · · · · · · · · · · · · · ·")
        lines.append(f"  {idx:<4} {combo:<6} {bar} {pct:>4.1f}%  {oran:>6.2f}")

    lines.append("━" * W)
    lines.append(f"")
    lines.append("━" * W)
    lines.append(f"  ✅ EN GÜÇLÜ 2 İHTİMAL")
    lines.append("━" * W)
    for i, (combo, prob) in enumerate(sorted_iyms[:2], 1):
        oran = (1 / prob) * PAYOUT
        emoji = "🥇" if i == 1 else "🥈"
        lines.append(f"  {emoji} {combo}  —  %{prob*100:.1f}  —  Oran: {oran:.2f}")
    lines.append("━" * W)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# GÖRSEL OKUMA (CLAUDE VİSİON API)
# ══════════════════════════════════════════════════════════════
VISION_PROMPT = """Bu istatistik.nesine.com ekran görüntüsünden futbol maç verilerini çıkar.

Sayfada 2 takım var: üstteki EV SAHİBİ, alttaki DEPLASMAN.

Her takım için:
- "Genel" sekmesinde son 6 maçın MS (maç sonu) ve İY (ilk yarı) skorlarını
- "İç & Dış" sekmesinde son 6 maçın MS ve İY skorlarını

Eğer görselde "Genel" sekmesi aktifse → genel veriler var
Eğer "İç & Dış" sekmesi aktifse → iç/dış saha verileri var

Takımın KENDI perspektifinden yaz: (takımın_attığı, takımın_yediği)

Ev sahibi için: iç sahada oynadığı maçlar (İç & Dış sekmesinde üstteki takım)
Deplasman için: dış sahada oynadığı maçlar (İç & Dış sekmesinde alttaki takım)

SADECE JSON döndür, başka hiçbir şey yazma:
{
  "home_team": "takım adı",
  "away_team": "takım adı",
  "home_genel": [[attığı, yediği], ...],
  "home_ic": [[attığı, yediği], ...],
  "away_genel": [[attığı, yediği], ...],
  "away_dis": [[attığı, yediği], ...]
}

Eğer görselde sadece Genel veya sadece İç&Dış varsa, mevcut olanı doldur, diğerini boş liste [] bırak.
Mutlaka en az 4, tercihen 6 maç çıkar."""

VISION_PROMPT_PAIR = """Bu iki görsel istatistik.nesine.com'dan alınmış:
- Görsel 1: Genel sekmesi (her iki takımın son 6 genel maçı)
- Görsel 2: İç & Dış sekmesi (ev sahibinin iç saha, deplasman takımının dış saha maçları)

Her takım için attığı ve yediği golleri takımın kendi perspektifinden çıkar.

SADECE JSON döndür:
{
  "home_team": "takım adı",
  "away_team": "takım adı", 
  "home_genel": [[attığı, yediği], ...],
  "home_ic": [[attığı, yediği], ...],
  "away_genel": [[attığı, yediği], ...],
  "away_dis": [[attığı, yediği], ...]
}"""


def extract_data_from_images(image_list):
    """
    image_list: [{"base64": "...", "media_type": "image/jpeg"}, ...]
    1 görsel: tek sekme analizi
    2 görsel: 1. genel, 2. iç/dış
    """
    # Grok vision için mesaj içeriği oluştur
    content = []

    if len(image_list) == 1:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{image_list[0]['media_type']};base64,{image_list[0]['base64']}"
            }
        })
        content.append({"type": "text", "text": VISION_PROMPT})
    else:
        for img in image_list[:2]:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img['media_type']};base64,{img['base64']}"
                }
            })
        content.append({"type": "text", "text": VISION_PROMPT_PAIR})

    response = client.chat.completions.create(
        model="grok-2-vision-latest",
        max_tokens=2000,
        messages=[{"role": "user", "content": content}]
    )

    raw = response.choices[0].message.content.strip()

    # JSON temizle
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    raw = raw.strip()

    data = json.loads(raw)
    return data


# ══════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        files = request.files.getlist('images')
        if not files or len(files) == 0:
            return jsonify({"error": "Görsel yüklenmedi"}), 400

        # Görselleri grupla: her maç için 1 veya 2 görsel
        # Eğer tek görsel → tek analiz
        # Eğer çift görsel → çiftler halinde analiz (genel + iç/dış)
        results = []
        
        # Görselleri base64'e çevir
        images = []
        for f in files:
            data = f.read()
            b64 = base64.standard_b64encode(data).decode('utf-8')
            mt = f.content_type or 'image/jpeg'
            images.append({"base64": b64, "media_type": mt, "name": f.filename})

        # Gruplama stratejisi:
        # Eğer 2'nin katı sayıda görsel varsa → çiftler (genel + iç/dış)
        # Değilse → tek tek
        total = len(images)
        
        if total % 2 == 0:
            # Çiftler halinde işle
            pairs = [(images[i], images[i+1]) for i in range(0, total, 2)]
            for pair in pairs:
                try:
                    data = extract_data_from_images(pair)
                    result_text = process_match_data(data)
                    results.append({
                        "success": True,
                        "match": f"{data.get('home_team','?')} vs {data.get('away_team','?')}",
                        "output": result_text
                    })
                except Exception as e:
                    results.append({
                        "success": False,
                        "match": f"Görsel {pair[0]['name']}",
                        "error": str(e)
                    })
        else:
            # Tek tek işle
            for img in images:
                try:
                    data = extract_data_from_images([img])
                    result_text = process_match_data(data)
                    results.append({
                        "success": True,
                        "match": f"{data.get('home_team','?')} vs {data.get('away_team','?')}",
                        "output": result_text
                    })
                except Exception as e:
                    results.append({
                        "success": False,
                        "match": img['name'],
                        "error": str(e)
                    })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def process_match_data(data):
    """Çıkarılan veriyi modele ver ve çıktı üret"""
    home_team  = data.get("home_team", "Ev Sahibi")
    away_team  = data.get("away_team", "Deplasman")
    home_genel = [tuple(x) for x in data.get("home_genel", [])]
    home_ic    = [tuple(x) for x in data.get("home_ic", [])]
    away_genel = [tuple(x) for x in data.get("away_genel", [])]
    away_dis   = [tuple(x) for x in data.get("away_dis", [])]

    # Eksik veri varsa genel ile doldur
    if not home_ic:
        home_ic = home_genel
    if not away_dis:
        away_dis = away_genel

    if len(home_genel) < 3 or len(away_genel) < 3:
        raise ValueError("Yeterli maç verisi çıkarılamadı")

    sorted_iyms, lam_h, lam_a = value_hunting_model(
        home_genel, home_ic, away_genel, away_dis
    )

    return build_output(home_team, away_team, sorted_iyms, lam_h, lam_a)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
