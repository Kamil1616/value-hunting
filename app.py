import os
import sys
import math
import base64
import json
import re
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from PIL import Image
import io

app = Flask(__name__)

client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

W_SPEC     = 0.55
W_GEN      = 0.45
HT_FACTOR  = 0.48
GS_LEADER  = 1.08
GS_DRAW    = 1.04
LEAGUE_AVG = 1.20
PAYOUT     = 0.90
DC_RHO     = -0.13

def poisson_pmf(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
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
    if not lst: return 1.0
    return sum(x[idx] for x in lst) / len(lst)

def value_hunting_model(home_genel, home_ic, away_genel, away_dis):
    h_att = W_SPEC * avg(home_ic, 0)  + W_GEN * avg(home_genel, 0)
    h_def = W_SPEC * avg(home_ic, 1)  + W_GEN * avg(home_genel, 1)
    a_att = W_SPEC * avg(away_dis, 0) + W_GEN * avg(away_genel, 0)
    a_def = W_SPEC * avg(away_dis, 1) + W_GEN * avg(away_genel, 1)
    lam_h = math.sqrt((h_att / LEAGUE_AVG) * (a_def / LEAGUE_AVG)) * LEAGUE_AVG
    lam_a = math.sqrt((a_att / LEAGUE_AVG) * (h_def / LEAGUE_AVG)) * LEAGUE_AVG
    score_probs = {}
    for i in range(7):
        for j in range(7):
            p = poisson_pmf(i, lam_h) * poisson_pmf(j, lam_a) * dixon_coles(i, j, lam_h, lam_a)
            score_probs[(i, j)] = p
    t = sum(score_probs.values())
    score_probs = {k: v/t for k,v in score_probs.items()}
    ht_probs = {}
    for i in range(5):
        for j in range(5):
            ht_probs[(i, j)] = poisson_pmf(i, lam_h*HT_FACTOR) * poisson_pmf(j, lam_a*HT_FACTOR)
    t2 = sum(ht_probs.values())
    ht_probs = {k: v/t2 for k,v in ht_probs.items()}
    iyms = {}
    for (hi,hj),hp in ht_probs.items():
        for (fi,fj),fp in score_probs.items():
            key = f"{result(hi,hj)}/{result(fi,fj)}"
            ht_r = result(hi,hj); ft_r = result(fi,fj)
            gs = GS_LEADER if (ht_r==ft_r and ht_r!='X') else (GS_DRAW if (ht_r=='X' and ft_r!='X') else 1.0)
            iyms[key] = iyms.get(key,0) + hp*fp*gs
    t3 = sum(iyms.values())
    iyms = {k: v/t3 for k,v in iyms.items()}
    return sorted(iyms.items(), key=lambda x: x[1], reverse=True), lam_h, lam_a

def format_bar(pct, max_len=20):
    filled = min(int(pct*max_len/35), max_len)
    return '▓'*filled + '░'*(max_len-filled)

def build_output(home_name, away_name, sorted_iyms, lam_h, lam_a):
    W = 52
    lines = ["━"*W, f"  MAÇ · {home_name} vs {away_name}", f"  λ_ev={lam_h:.3f}  |  λ_dep={lam_a:.3f}", "━"*W,
             f"  {'#':<4} {'İY/MS':<6} {'OLASILIK':<24} {'ORAN':>6}", "━"*W]
    for idx,(combo,prob) in enumerate(sorted_iyms,1):
        oran = (1/prob)*PAYOUT
        pct = prob*100
        if idx==6: lines.append(f"  · · · · · · · · · · · · · · · · · · · · · · · ·")
        lines.append(f"  {idx:<4} {combo:<6} {format_bar(pct)} {pct:>4.1f}%  {oran:>6.2f}")
    lines += ["━"*W, "", "━"*W, "  ✅ EN GÜÇLÜ 2 İHTİMAL", "━"*W]
    for i,(combo,prob) in enumerate(sorted_iyms[:2],1):
        oran = (1/prob)*PAYOUT
        lines.append(f"  {'🥇' if i==1 else '🥈'} {combo}  —  %{prob*100:.1f}  —  Oran: {oran:.2f}")
    lines.append("━"*W)
    return "\n".join(lines)

VISION_PROMPT = """Bu iki görsel istatistik.nesine.com'dan:
- Görsel 1: Genel sekmesi
- Görsel 2: İç & Dış sekmesi

Takımın KENDI perspektifinden: (attığı, yediği)

SADECE JSON döndür:
{
  "home_team": "takım adı",
  "away_team": "takım adı",
  "home_genel": [[attığı, yediği], ...],
  "home_ic": [[attığı, yediği], ...],
  "away_genel": [[attığı, yediği], ...],
  "away_dis": [[attığı, yediği], ...]
}"""

def compress_image(data):
    try:
        img = Image.open(io.BytesIO(data))
        img.thumbnail((600, 600), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=60)
        return buf.getvalue()
    except:
        return data

def extract_data_from_images(img1, img2):
    content = []
    for img_data in [img1, img2]:
        compressed = compress_image(img_data)
        b64 = base64.standard_b64encode(compressed).decode('utf-8')
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": VISION_PROMPT})

    response = client.chat.completions.create(
        model="qwen/qwen3-vl-30b-a3b-thinking",
        max_tokens=1000,
        timeout=60,
        messages=[{"role": "user", "content": content}]
    )
    raw = response.choices[0].message.content.strip()
    print(f"[RAW]: {raw[:200]}", file=sys.stderr, flush=True)
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    raw = raw.strip()
    return json.loads(raw)

def process_match_data(data):
    home_team  = data.get("home_team", "Ev Sahibi")
    away_team  = data.get("away_team", "Deplasman")
    home_genel = [tuple(x) for x in data.get("home_genel", [])]
    home_ic    = [tuple(x) for x in data.get("home_ic", [])]
    away_genel = [tuple(x) for x in data.get("away_genel", [])]
    away_dis   = [tuple(x) for x in data.get("away_dis", [])]
    if not home_ic: home_ic = home_genel
    if not away_dis: away_dis = away_genel
    if len(home_genel) < 3 or len(away_genel) < 3:
        raise ValueError("Yeterli veri çıkarılamadı")
    sorted_iyms, lam_h, lam_a = value_hunting_model(home_genel, home_ic, away_genel, away_dis)
    return build_output(home_team, away_team, sorted_iyms, lam_h, lam_a)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        files = request.files.getlist('images')
        if len(files) < 2:
            return jsonify({"error": "2 görsel gerekli (Genel + İç/Dış)"}), 400
        img1 = files[0].read()
        img2 = files[1].read()
        data = extract_data_from_images(img1, img2)
        output = process_match_data(data)
        return jsonify({"results": [{"success": True, "match": f"{data.get('home_team','?')} vs {data.get('away_team','?')}", "output": output}]})
    except Exception as e:
        print(f"[HATA]: {str(e)}", file=sys.stderr, flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
