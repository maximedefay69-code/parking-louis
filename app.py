import streamlit as st
import requests, pytz, joblib, math, numpy as np, pandas as pd, re, gspread, json
from datetime import datetime
from google.oauth2.service_account import Credentials

# --- 1. DONNÉES SOCIO-ÉCO (INCHANGÉES) ---
DATA_ARRDT = {
    1: {"REV_M": 2916, "VEH": 0.32}, 2: {"REV_M": 2666, "VEH": 0.28}, 3: {"REV_M": 2833, "VEH": 0.25},
    4: {"REV_M": 2750, "VEH": 0.26}, 5: {"REV_M": 3166, "VEH": 0.35}, 6: {"REV_M": 3750, "VEH": 0.38},
    7: {"REV_M": 4166, "VEH": 0.42}, 8: {"REV_M": 4000, "VEH": 0.40}, 9: {"REV_M": 3000, "VEH": 0.30},
    10: {"REV_M": 2333, "VEH": 0.22}, 11: {"REV_M": 2583, "VEH": 0.24}, 12: {"REV_M": 2500, "VEH": 0.33},
    13: {"REV_M": 2250, "VEH": 0.35}, 14: {"REV_M": 2583, "VEH": 0.36}, 15: {"REV_M": 2916, "VEH": 0.38},
    16: {"REV_M": 4583, "VEH": 0.45}, 17: {"REV_M": 3083, "VEH": 0.37}, 18: {"REV_M": 2083, "VEH": 0.20},
    19: {"REV_M": 1833, "VEH": 0.22}, 20: {"REV_M": 1916, "VEH": 0.23}
}

# --- 2. FONCTIONS API AMÉLIORÉES ---
def obtenir_places_chirurgical(type_saisi, nom_saisi, arrondissement):
    try:
        nom_cherche = nom_saisi.upper().strip()
        url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        
        # On filtre par arrondissement directement dans la requête API pour plus de rapidité
        params = {
            "where": f"suggest(nomvoie, '{nom_cherche}') AND arrond = {arrondissement}",
            "limit": 100
        }
        
        regimes_valides = ["PAYANT ROTATIF", "PAYANT MIXTE", "GRATUIT"]
        res = requests.get(url, params=params, timeout=10).json()
        
        total = 0
        voies_confirmees = set()
        
        if res and 'results' in res:
            for item in res['results']:
                # On recréé le nom complet comme dans l'API (typevoie + nomvoie)
                t_api = str(item.get('typevoie', '')).upper()
                n_api = str(item.get('nomvoie', '')).upper()
                nom_complet_api = f"{t_api} {n_api}"
                
                # Vérification : le nom saisi est dans le nomvoie ET le type correspond (RUE, AV, BD...)
                if nom_cherche in n_api:
                    # On est flexible sur le type (AV vs AVENUE)
                    type_match = False
                    if type_saisi.upper() == "BOULEVARD" and ("BD" in t_api or "BOULEVARD" in t_api): type_match = True
                    elif type_saisi.upper() == "AVENUE" and ("AV" in t_api or "AVENUE" in t_api): type_match = True
                    elif type_saisi.upper() in t_api: type_match = True
                    
                    if type_match:
                        regime = str(item.get('regpri', '')).upper()
                        if any(r in regime for r in regimes_valides):
                            total += item.get('placal', 0)
                            voies_confirmees.add(nom_complet_api)
        
        return total, list(voies_confirmees)
    except:
        return 15, ["Erreur API"]

# --- 3. LOGIQUE METEO / SHEETS (INCHANGÉE) ---
def save_to_google_sheets(data_row):
    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds_dict = json.loads(st.secrets["gcp_service_account"]["json_data"])
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(credentials)
        sheet = client.open("SLOT_Beta1").sheet1
        sheet.append_row(data_row)
        return True
    except: return False

def get_live_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code"
        res = requests.get(url, timeout=5).json()
        code = res['current']['weather_code']
        mto = "Beau" if code in [0, 1] else ("Nuageux" if code in [2, 3, 45, 48] else "Pluie")
        return mto, res['current']['temperature_2m']
    except: return "Beau", 18.0

@st.cache_resource
def load_assets():
    try: return joblib.load("modele_lightgbmDA.pkl"), joblib.load("preprocessorDA.pkl")
    except: return None, None

model, prepro = load_assets()

# --- 4. INTERFACE ---
st.set_page_config(page_title="IA Parking Paris", page_icon="🅿️")
st.title("🅿️ Collecte Terrain V59")

c1, c2, c3 = st.columns([1, 1.5, 2.5])
num_voie = c1.text_input("N°")
type_voie = c2.selectbox("Type", ["Rue", "Boulevard", "Avenue", "Place", "Quai", "Impasse"])
nom_rue = c3.text_input("Nom (ex: Rivoli)")

if st.button("🔍 ANALYSER LA ZONE"):
    if not nom_rue:
        st.warning("Précisez la rue.")
    else:
        with st.spinner('Géolocalisation...'):
            query = f"{num_voie} {type_voie} {nom_rue} Paris"
            geo_res = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={query.replace(' ', '+')}&limit=1").json()
            
            if 'features' in geo_res and len(geo_res['features']) > 0:
                feat = geo_res['features'][0]
                lat, lon = feat['geometry']['coordinates'][1], feat['geometry']['coordinates'][0]
                arrdt = int(feat['properties']['postcode'][-2:])
                
                # --- NOUVELLE LOGIQUE CHIRURGICALE ---
                # On passe l'arrondissement trouvé par le GPS à l'API de stationnement
                nb_places, voies = obtenir_places_chirurgical(type_voie, nom_rue, arrdt)
                
                socio = DATA_ARRDT.get(arrdt, {"REV_M": 2500, "VEH": 0.30})
                mto, temp = get_live_weather(lat, lon)
                
                # Trafic
                score_t = 0
                if st.secrets.get("GOOGLE_API_KEY"):
                    try:
                        t_url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={lat},{lon}&destinations={lat+0.001},{lon+0.001}&departure_time=now&key={st.secrets['GOOGLE_API_KEY']}"
                        r = requests.get(t_url).json()
                        diff = r['rows'][0]['elements'][0]['duration_in_traffic']['value'] - r['rows'][0]['elements'][0]['duration']['value']
                        score_t = 100 if diff > 120 else (50 if diff > 60 else 0)
                    except: pass

                # IA
                now = datetime.now(pytz.timezone('Europe/Paris'))
                X = pd.DataFrame([{
                    'RUE': nom_rue.upper(), 'VILLE': 'Paris', 'JOUR': now.strftime("%A"), 
                    'MTO': mto, 'TRAFIC': score_t, '% PARKING OC': 0.50, 
                    'NBR PLACES': nb_places, 'REVENUS / H': socio["REV_M"], 
                    'VEHICULES / H': socio["VEH"], 'TEMPERATURE': temp,
                    'HEURE_SIN': np.sin(2*np.pi*(now.hour*60+now.minute)/1440),
                    'HEURE_COS': np.cos(2*np.pi*(now.hour*60+now.minute)/1440)
                }])
                
                occ = model.predict(prepro.transform(X))[0]
                libres = max(0, math.floor(nb_places * (1 - occ)))

                st.info(f"📍 Arrondissement détecté : {arrdt}e")
                st.success(f"✅ {nb_places} places trouvées (API Paris) sur : {', '.join(voies)}")
                st.metric("Estimation places libres", f"{libres} PLACES")
                
                st.session_state['releve'] = {
                    "Date": now.strftime("%d/%m/%Y"), "Heure": now.strftime("%H:%M"),
                    "Num": num_voie, "Rue": f"{type_voie} {nom_rue}", "Arrdt": arrdt,
                    "Pred_P": libres, "Pred_O": f"{round(occ*100)}%",
                    "Mto": mto, "Temp": f"{temp}°C", "Trafic": score_t, "Total": nb_places
                }
            else:
                st.error("Adresse non trouvée.")

if 'releve' in st.session_state:
    st.divider()
    reel = st.number_input("Places RÉELLES libres ?", min_value=0, step=1)
    if st.button("💾 SAUVEGARDER"):
        d = st.session_state['releve']
        if save_to_google_sheets([d["Date"], d["Heure"], d["Num"], d["Rue"], d["Arrdt"], d["Pred_P"], d["Pred_O"], reel, d["Mto"], d["Temp"], d["Trafic"], d["Total"]]):
            st.balloons(); st.success("Données envoyées !"); del st.session_state['releve']
