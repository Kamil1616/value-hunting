import os
import sys
import math
import json
import re
import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

API_KEY = os.environ.get("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

W_SPEC     = 0.55
W_GEN      = 0.45
HT_FACTOR  = 0.48
GS_LEADER  = 1.08
GS_DRAW    = 1.04
LEAGUE_AVG = 1.20
PAYOUT     = 0.90
DC_RHO     = -0.14

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
    lines = ["━"*W, f"  MAÇ · {home_name} vs {away_name}",
             f"  λ_ev={lam_h:.3f}  |  λ_dep={lam_a:.3f}", "━"*W,
             f"  {'#':<4} {'İY/MS':<6} {'OLASILIK':<24} {'ORAN':>6}", "━"*W]
    for idx,(combo,prob) in enumerate(sorted_iyms,1):
        oran = (1/prob)*PAYOUT
        pct = prob*100
        if idx==6: lines.append("  · · · · · · · · · · · · · · · · · · · · · · · ·")
        lines.append(f"  {idx:<4} {combo:<6} {format_bar(pct)} {pct:>4.1f}%  {oran:>6.2f}")
    lines += ["━"*W, "", "━"*W, "  ✅ EN GÜÇLÜ 2 İHTİMAL", "━"*W]
    for i,(combo,prob) in enumerate(sorted_iyms[:2],1):
        oran = (1/prob)*PAYOUT
        lines.append(f"  {'🥇' if i==1 else '🥈'} {combo}  —  %{prob*100:.1f}  —  Oran: {oran:.2f}")
    lines.append("━"*W)
    return "\n".join(lines)

def get_team_id(team_name):
    r = requests.get(f"{BASE_URL}/teams", headers=HEADERS, params={"search": team_name})
    data = r.json()
    teams = data.get("response", [])
    if not teams:
        raise ValueError(f"Takım bulunamadı: {team_name}")
    return teams[0]["team"]["id"], teams[0]["team"]["name"]

def get_last_matches(team_id, venue, n=6):
    """venue: home veya away"""
    r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params={
        "team": team_id,
        "last": 20,
        "status": "FT"
    })
    fixtures = r.json().get("response", [])
    matches = []
    for f in fixtures:
        home_id = f["teams"]["home"]["id"]
        away_id = f["teams"]["away"]["id"]
        hg = f["goals"]["home"]
        ag = f["goals"]["away"]
        if hg is None or ag is None:
            continue
        if venue == "home" and home_id == team_id:
            matches.append((hg, ag))
        elif venue == "away" and away_id == team_id:
            matches.append((ag, hg))
        if len(matches) >= n:
            break
    return matches

def get_all_matches(team_id, n=6):
    r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params={
        "team": team_id,
        "last": 20,
        "status": "FT"
    })
    fixtures = r.json().get("response", [])
    matches = []
    for f in fixtures:
        home_id = f["teams"]["home"]["id"]
        hg = f["goals"]["home"]
        ag = f["goals"]["away"]
        if hg is None or ag is None:
            continue
        if home_id == team_id:
            matches.append((hg, ag))
        else:
            matches.append((ag, hg))
        if len(matches) >= n:
            break
    return matches

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    try:
        r = requests.get(f"{BASE_URL}/status", headers=HEADERS)
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        home_name = data.get("home_team", "").strip()
        away_name = data.get("away_team", "").strip()
        if not home_name or not away_name:
            return jsonify({"error": "Takım adları gerekli"}), 400

        home_id, home_full = get_team_id(home_name)
        away_id, away_full = get_team_id(away_name)

        home_genel = get_all_matches(home_id)
        home_ic    = get_last_matches(home_id, "home")
        away_genel = get_all_matches(away_id)
        away_dis   = get_last_matches(away_id, "away")

        if not home_ic: home_ic = home_genel
        if not away_dis: away_dis = away_genel

        if len(home_genel) < 3 or len(away_genel) < 3:
            return jsonify({"error": "Yeterli maç verisi bulunamadı"}), 400

        sorted_iyms, lam_h, lam_a = value_hunting_model(home_genel, home_ic, away_genel, away_dis)
        output = build_output(home_full, away_full, sorted_iyms, lam_h, lam_a)

        return jsonify({"success": True, "output": output,
                        "home_matches": len(home_genel), "away_matches": len(away_genel)})
    except Exception as e:
        print(f"[HATA]: {str(e)}", file=sys.stderr, flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
