import streamlit as st
import requests, pytz, joblib, math, numpy as np, pandas as pd, json
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

# --- 1. CONNEXION VIA SECRETS ---
def save_to_google_sheets(data_row):
    try:
        raw_json = st.secrets["gcp_service_account"]["json_data"]
        info = json.loads(raw_json)
        
        creds = Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        )
        
        client = gspread.authorize(creds)
        sheet = client.open("SLOT_Beta1").sheet1
        sheet.append_row(data_row)
        return True
    except Exception as e:
        st.error(f"❌ Erreur de connexion Google Sheets : {e}")
        return False

# --- 2. CONFIGURATION ---
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

# --- 3. FONCTIONS DYNAMIQUES ---
def obtenir_places_total(type_v, nom_v, arrdt):
    try:
        nom_c = nom_v.upper().strip()
        url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        params = {"where": f"suggest(nomvoie, '{nom_c}') AND arrond = {arrdt}", "limit": 100}
        res = requests.get(url, params=params, timeout=10).json()
        s, v = 0, []
        if 'results' in res:
            for i in res['results']:
                t_api = str(i.get('typevoie','')).upper()
                n_api = str(i.get('nomvoie','')).upper()
                reg = i.get('regpri', '')
                if reg is None: reg = ""
                reg = str(reg).upper()
                p = i.get('placal', 0)
                
                match = False
                tl = type_v.upper()
                if tl == "BOULEVARD" and ("BD" in t_api or "BOULEVARD" in t_api): match = True
                elif tl == "AVENUE" and ("AV" in t_api or "AVENUE" in t_api): match = True
                elif tl in t_api: match = True
                
                if nom_c in n_api and match:
                    if any(r in reg for r in ["PAYANT ROTATIF", "PAYANT MIXTE", "GRATUIT"]):
                        s += p
                        v.append(f"{t_api} {n_api}")
        return s, list(set(v))
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

# --- 4. INTERFACE ---
st.set_page_config(page_title="SLOT V70 - Benchmark", layout="wide")
st.title("🅿️ SLOT - Assistant Terrain")

c1, c2, c3 = st.columns([1, 2, 3])
num_v = c1.text_input("N°")
type_v = c2.selectbox("Type", ["Rue", "Boulevard", "Avenue", "Place", "Quai", "Impasse"])
nom_v = c3.text_input("Nom de la rue")

if st.button("🚀 ANALYSER"):
    if nom_v:
        with st.spinner("Analyse..."):
            geo = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={num_v}+{type_v}+{nom_v}+Paris&limit=1").json()
            if 'features' in geo and len(geo['features']) > 0:
                f = geo['features'][0]
                lat, lon = f['geometry']['coordinates'][1], f['geometry']['coordinates'][0]
                arrdt = int(f['properties']['postcode'][-2:])
                
                total_p, liste_v = obtenir_places_total(type_v, nom_v, arrdt)
                mto, temp = get_weather(lat, lon)
                socio = DATA_ARRDT.get(arrdt, {"REV_M": 2500, "VEH": 0.30})
                
                now = datetime.now(pytz.timezone('Europe/Paris'))
                minutes = now.hour * 60 + now.minute
                
                st.subheader("📊 Panel de Contrôle")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Météo", mto)
                k2.metric("Temp", f"{temp}°C")
                k3.metric("Trafic", "0.0")
                k4.metric("Places", total_p)

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
                    'MTO': mto,
                    'TEMPERATURE': temp,
                    'HEURE_MINUTES': minutes,
                    'HEURE_SIN': np.sin(2 * np.pi * minutes / 1440),
                    'HEURE_COS': np.cos(2 * np.pi * minutes / 1440)
                }
                
                X_df = pd.DataFrame([X_dict])[['DATE','JOUR','HEURE','RUE','VILLE','TRAFIC','% PARKING OC','NBR PLACES','REVENUS / H','VEHICULES / H','MTO','TEMPERATURE','HEURE_MINUTES','HEURE_SIN','HEURE_COS']]

                try:
                    occ = model.predict(prepro.transform(X_df))[0]
                    libres = max(0, math.floor(total_p * (1 - occ)))
                    st.divider()
                    st.success(f"🤖 IA : **{libres} places libres**.")
                    
                    st.session_state['save'] = [
                        now.strftime("%d/%m/%Y"), now.strftime("%H:%M"), f"{num_v} {type_v} {nom_v}",
                        arrdt, libres, mto, temp, total_p, f"{round(occ*100)}%"
                    ]
                except Exception as e: st.error(f"Erreur IA : {e}")

# --- 5. ENREGISTREMENT & BENCHMARK ---
if 'save' in st.session_state:
    st.divider()
    st.subheader("🏁 Benchmark Terrain")
    col_v1, col_v2, col_v3 = st.columns([1, 1, 2])
    
    reel = col_v1.number_input("Places réelles :", min_value=0, step=1)
    
    # AJOUT DU MENU PARKNAV
    parknav = col_v2.selectbox("Parknav :", ["Vert", "Orange", "Rouge", "Gris (N/C)"])
    
    note = col_v3.text_input("Note libre")
    
    if st.button("💾 ENVOYER AU SHEETS"):
        # On ajoute reel, parknav et note à la liste existante dans session_state
        if save_to_google_sheets(st.session_state['save'] + [reel, parknav, note]):
            st.balloons()
            st.success("✅ Données enregistrées dans SLOT_Beta1 !")
            del st.session_state['save']
            st.rerun()
