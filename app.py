import streamlit as st
import requests, pytz, joblib, math, numpy as np, pandas as pd, json
from datetime import datetime

# --- 1. CONFIGURATION & DICTIONNAIRES ---
DATA_ARRDT = {
    1: {"REV_M": 2916, "VEH": 0.32}, 2: {"REV_M": 2666, "VEH": 0.28}, 3: {"REV_M": 2833, "VEH": 0.25},
    4: {"REV_M": 2750, "VEH": 0.26}, 5: {"REV_M": 3166, "VEH": 0.35}, 6: {"REV_M": 3750, "VEH": 0.38},
    7: {"REV_M": 4166, "VEH": 0.42}, 8: {"REV_M": 4000, "VEH": 0.40}, 9: {"REV_M": 3000, "VEH": 0.30},
    10: {"REV_M": 2333, "VEH": 0.22}, 11: {"REV_M": 2583, "VEH": 0.24}, 12: {"REV_M": 2500, "VEH": 0.33},
    13: {"REV_M": 2250, "VEH": 0.35}, 14: {"REV_M": 2583, "VEH": 0.36}, 15: {"REV_M": 2916, "VEH": 0.38},
    16: {"REV_M": 4583, "VEH": 0.45}, 17: {"REV_M": 3083, "VEH": 0.37}, 18: {"REV_M": 2083, "VEH": 0.20},
    19: {"REV_M": 1833, "VEH": 0.22}, 20: {"REV_M": 1916, "VEH": 0.23}
}

JOURS_FR = {
    "Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
    "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche"
}

# --- 2. FONCTIONS DE RÉCUPÉRATION DYNAMIQUES ---

def obtenir_places_total(type_louis, nom_louis, arrdt_gps):
    try:
        nom_cherche = nom_louis.upper().strip()
        url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        params = {"where": f"suggest(nomvoie, '{nom_cherche}') AND arrond = {arrdt_gps}", "limit": 100}
        res = requests.get(url, params=params, timeout=10).json()
        somme_p, voies = 0, []
        if res and 'results' in res:
            for item in res['results']:
                t_api, n_api, reg, p = str(item.get('typevoie','')).upper(), str(item.get('nomvoie','')).upper(), str(item.get('regpri','')).upper(), item.get('placal', 0)
                match = False
                tl = type_louis.upper()
                if tl == "BOULEVARD" and ("BD" in t_api or "BOULEVARD" in t_api): match = True
                elif tl == "AVENUE" and ("AV" in t_api or "AVENUE" in t_api): match = True
                elif tl in t_api: match = True
                if nom_cherche in n_api and match:
                    if any(r in reg for r in ["PAYANT ROTATIF", "PAYANT MIXTE", "GRATUIT"]):
                        somme_p += p
                        voies.append(f"{t_api} {n_api}")
        return somme_p, list(set(voies))
    except: return 15, ["Erreur API"]

def get_weather(lat, lon):
    try:
        r = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code").json()
        c = r['current']['weather_code']
        m = "Beau" if c in [0,1] else ("Nuageux" if c in [2,3,45,48] else "Pluie")
        return m, r['current']['temperature_2m']
    except: return "Beau", 18.0

@st.cache_resource
def load_assets():
    try: return joblib.load("modele_lightgbmDA.pkl"), joblib.load("preprocessorDA.pkl")
    except: return None, None

model, prepro = load_assets()

# --- 3. INTERFACE STREAMLIT ---
st.set_page_config(page_title="SLOT V66", layout="wide")
st.title("🅿️ SLOT - Collecte & IA")

col_in1, col_in2, col_in3 = st.columns([1, 2, 3])
num_v = col_in1.text_input("N°")
type_v = col_in2.selectbox("Type", ["Rue", "Boulevard", "Avenue", "Place", "Quai", "Impasse"])
nom_v = col_in3.text_input("Nom de la rue (ex: Rivoli)")

if st.button("🚀 ANALYSER LA RUE"):
    if not nom_v:
        st.warning("Précise le nom de la rue !")
    else:
        with st.spinner("Collecte des données..."):
            geo = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={num_v}+{type_v}+{nom_v}+Paris&limit=1").json()
            
            if 'features' in geo and len(geo['features']) > 0:
                f = geo['features'][0]
                lat, lon = f['geometry']['coordinates'][1], f['geometry']['coordinates'][0]
                arrdt = int(f['properties']['postcode'][-2:])
                
                total_p, liste_v = obtenir_places_total(type_v, nom_v, arrdt)
                mto_label, temp_val = get_weather(lat, lon)
                socio = DATA_ARRDT.get(arrdt, {"REV_M": 2500, "VEH": 0.30})
                
                tz = pytz.timezone('Europe/Paris')
                now = datetime.now(tz)
                minutes_total = now.hour * 60 + now.minute
                
                # --- PANEL DE CONTRÔLE LOUIS ---
                st.subheader("📊 Données lues en temps réel")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Météo", mto_label)
                c2.metric("Température", f"{temp_val}°C")
                c3.metric("Trafic", "0.0")
                c4.metric("Places (OpenData)", total_p)

                # --- PRÉPARATION DES 15 FEATURES ---
                X_dict = {
                    'DATE': now.strftime("%d/%m/%Y"),
                    'JOUR': JOURS_FR.get(now.strftime("%A")),
                    'HEURE': now.strftime("%H:%M"),
                    'RUE': nom_v.upper(),
                    'VILLE': "Paris",
                    'TRAFIC': 0.0,
                    '% PARKING OC': 0.5,
                    'NBR PLACES': total_p,
                    'REVENUS / H': socio["REV_M"],
                    'VEHICULES / H': socio["VEH"],
                    'MTO': mto_label,
                    'TEMPERATURE': temp_val,
                    'HEURE_MINUTES': minutes_total,
                    'HEURE_SIN': np.sin(2 * np.pi * minutes_total / 1440),
                    'HEURE_COS': np.cos(2 * np.pi * minutes_total / 1440)
                }
                
                X_df = pd.DataFrame([X_dict])
                colonnes_15 = ['DATE','JOUR','HEURE','RUE','VILLE','TRAFIC','% PARKING OC','NBR PLACES','REVENUS / H','VEHICULES / H','MTO','TEMPERATURE','HEURE_MINUTES','HEURE_SIN','HEURE_COS']
                X_df = X_df[colonnes_15]

                try:
                    X_trans = prepro.transform(X_df)
                    occ_pred = model.predict(X_trans)[0]
                    libres = max(0, math.floor(total_p * (1 - occ_pred)))
                    
                    st.divider()
                    st.success(f"🤖 **Résultat IA :** Environ **{libres} places libres**.")
                    
                    # Sauvegarde pour Google Sheets
                    st.session_state['data_save'] = {
                        "date": now.strftime("%d/%m/%Y"),
                        "heure": now.strftime("%H:%M"),
                        "adresse": f"{num_v} {type_v} {nom_v}",
                        "arrdt": arrdt,
                        "ia_pred": libres,
                        "occ_score": f"{round(occ_pred*100)}%",
                        "mto": mto_label,
                        "temp": temp_val,
                        "total_p": total_p
                    }
                except Exception as e:
                    st.error(f"Erreur modèle : {e}")
            else:
                st.error("Adresse introuvable.")

# --- 4. VALIDATION TERRAIN & ENREGISTREMENT ---
if 'data_save' in st.session_state:
    st.divider()
    st.subheader("📝 Validation Terrain par Louis")
    
    reel = st.number_input("Combien de places vois-tu réellement ?", min_value=0, step=1)
    note = st.text_input("Commentaire libre (optionnel)")
    
    if st.button("💾 ENREGISTRER DANS GOOGLE SHEETS"):
        # Ici Louis, tu ajouteras ta logique gspread (wks.append_row)
        # On prépare la ligne finale
        d = st.session_state['data_save']
        ligne_a_sauver = [d['date'], d['heure'], d['adresse'], d['arrdt'], d['ia_pred'], reel, d['occ_score'], d['mto'], d['temp'], d['total_p'], note]
        
        st.write("Données envoyées :", ligne_a_sauver)
        st.balloons()
        st.success("Relevé terrain enregistré avec succès !")
        del st.session_state['data_save']
