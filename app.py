import streamlit as st
import requests, pytz, joblib, math, numpy as np, pandas as pd, re, gspread, json
from datetime import datetime
from google.oauth2.service_account import Credentials

# --- DONNÉES SOCIO-ÉCO PAR ARRONDISSEMENT (TES DONNÉES) ---
DATA_ARRDT = {
    1: {"REV_M": 2916, "VEH": 0.32}, 2: {"REV_M": 2666, "VEH": 0.28}, 3: {"REV_M": 2833, "VEH": 0.25},
    4: {"REV_M": 2750, "VEH": 0.26}, 5: {"REV_M": 3166, "VEH": 0.35}, 6: {"REV_M": 3750, "VEH": 0.38},
    7: {"REV_M": 4166, "VEH": 0.42}, 8: {"REV_M": 4000, "VEH": 0.40}, 9: {"REV_M": 3000, "VEH": 0.30},
    10: {"REV_M": 2333, "VEH": 0.22}, 11: {"REV_M": 2583, "VEH": 0.24}, 12: {"REV_M": 2500, "VEH": 0.33},
    13: {"REV_M": 2250, "VEH": 0.35}, 14: {"REV_M": 2583, "VEH": 0.36}, 15: {"REV_M": 2916, "VEH": 0.38},
    16: {"REV_M": 4583, "VEH": 0.45}, 17: {"REV_M": 3083, "VEH": 0.37}, 18: {"REV_M": 2083, "VEH": 0.20},
    19: {"REV_M": 1833, "VEH": 0.22}, 20: {"REV_M": 1916, "VEH": 0.23}
}

# --- CONFIGURATION GOOGLE SHEETS (SOLUTION 2) ---
def save_to_google_sheets(data_row):
    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        json_info = st.secrets["gcp_service_account"]["json_data"]
        creds_dict = json.loads(json_info)
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(credentials)
        sheet = client.open("SLOT_Beta1").sheet1
        sheet.append_row(data_row)
        return True
    except Exception as e:
        st.error(f"Erreur Google Sheets : {e}")
        return False

# --- FONCTION MÉTÉO ---
def get_live_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code"
        res = requests.get(url, timeout=5).json()
        temp = res['current']['temperature_2m']
        code = res['current']['weather_code']
        mto = "Beau" if code in [0, 1] else ("Nuageux" if code in [2, 3, 45, 48] else "Pluie")
        return mto, temp
    except: return "Beau", 18.0

# --- NETTOYAGE RUE ---
def extraire_nom_propre(adresse):
    s = adresse.upper()
    s = re.sub(r'\d+', '', s)
    mots_a_supprimer = ["RUE DU ", "RUE DES ", "RUE DE LA ", "RUE DE ", "RUE ", "AVENUE ", "BOULEVARD ", "PLACE ", "D'"]
    for mot in mots_a_supprimer: s = s.replace(mot, "")
    return s.strip()

@st.cache_resource
def load_assets():
    try:
        return joblib.load("modele_lightgbmDA.pkl"), joblib.load("preprocessorDA.pkl")
    except: return None, None

model, prepro = load_assets()

# --- INTERFACE ---
st.set_page_config(page_title="IA Parking Paris - Collecte", page_icon="🅿️")
st.title("🅿️ IA Parking Paris (Mode Collecte)")

col_n, col_r = st.columns([1, 3])
numero = col_n.text_input("N°")
rue_saisie = col_r.text_input("Nom de la rue")

if st.button("1. ANALYSER LA ZONE"):
    if rue_saisie:
        with st.spinner('Analyse en cours...'):
            geo = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={rue_saisie}+Paris&limit=1").json()
            if geo['features']:
                feat = geo['features'][0]
                lat, lon = feat['geometry']['coordinates'][1], feat['geometry']['coordinates'][0]
                arrdt = int(feat['properties']['postcode'][-2:])
                nom_pur = extraire_nom_propre(rue_saisie)
                
                # --- RÉCUPÉRATION DES DONNÉES SOCIO-ÉCO ---
                # On utilise tes DATA_ARRDT ou des valeurs par défaut si l'arrondissement est inconnu
                socio = DATA_ARRDT.get(arrdt, {"REV_M": 2500, "VEH": 0.30})
                
                mto_live, temp_live = get_live_weather(lat, lon)
                
                # Trafic
                score_trafic = 0
                API_MAPS = st.secrets.get("GOOGLE_API_KEY")
                if API_MAPS:
                    try:
                        t_url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={lat},{lon}&destinations={lat+0.001},{lon+0.001}&departure_time=now&key={API_MAPS}"
                        t_res = requests.get(t_url).json()
                        e = t_res['rows'][0]['elements'][0]
                        score_trafic = 100 if (e['duration_in_traffic']['value'] - e['duration']['value']) > 120 else 50
                    except: pass

                # --- PRÉDICTION IA ---
                now = datetime.now(pytz.timezone('Europe/Paris'))
                X = pd.DataFrame([{
                    'RUE': nom_pur, 'VILLE': 'Paris', 'JOUR': now.strftime("%A"), 
                    'MTO': mto_live, 'TRAFIC': score_trafic, '% PARKING OC': 0.50, 
                    'NBR PLACES': 15, 
                    'REVENUS / H': socio["REV_M"], 
                    'VEHICULES / H': socio["VEH"],
                    'TEMPERATURE': temp_live,
                    'HEURE_SIN': np.sin(2*np.pi*(now.hour*60+now.minute)/1440),
                    'HEURE_COS': np.cos(2*np.pi*(now.hour*60+now.minute)/1440)
                }])
                
                occ_pred = model.predict(prepro.transform(X))[0]
                pred_places = max(0, math.floor(15 * (1 - occ_pred)))

                st.session_state['releve'] = {
                    "Date": now.strftime("%d/%m/%Y"), "Heure": now.strftime("%H:%M"),
                    "Num": numero, "Rue": nom_pur, "Arrdt": arrdt,
                    "Pred_P": pred_places, "Pred_O": f"{round(occ_pred*100)}%",
                    "Mto": mto_live, "Temp": f"{temp_live}°C", "Trafic": score_trafic
                }

                st.success(f"Prédiction : {pred_places} places ({round(occ_pred*100)}% d'occ.)")
                st.info(f"Données utilisées : Revenu {socio['REV_M']}€ | Flux {socio['VEH']}")
            else: st.error("Adresse non trouvée.")

if 'releve' in st.session_state:
    st.divider()
    reel = st.number_input("Places réelles sur le terrain ?", min_value=0, step=1)
    if st.button("💾 ENREGISTRER"):
        d = st.session_state['releve']
        if save_to_google_sheets([d["Date"], d["Heure"], d["Num"], d["Rue"], d["Arrdt"], d["Pred_P"], d["Pred_O"], reel, d["Mto"], d["Temp"], d["Trafic"]]):
            st.balloons()
            st.success("Enregistré dans SLOT_Beta1 !")
            del st.session_state['releve']
