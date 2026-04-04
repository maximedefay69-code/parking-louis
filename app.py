import streamlit as st
import requests, pytz, joblib, math, numpy as np, pandas as pd, re, gspread, json
from datetime import datetime
from google.oauth2.service_account import Credentials

# --- 1. TES DONNÉES SOCIO-ÉCO PAR ARRONDISSEMENT (INCHANGÉES) ---
DATA_ARRDT = {
    1: {"REV_M": 2916, "VEH": 0.32}, 2: {"REV_M": 2666, "VEH": 0.28}, 3: {"REV_M": 2833, "VEH": 0.25},
    4: {"REV_M": 2750, "VEH": 0.26}, 5: {"REV_M": 3166, "VEH": 0.35}, 6: {"REV_M": 3750, "VEH": 0.38},
    7: {"REV_M": 4166, "VEH": 0.42}, 8: {"REV_M": 4000, "VEH": 0.40}, 9: {"REV_M": 3000, "VEH": 0.30},
    10: {"REV_M": 2333, "VEH": 0.22}, 11: {"REV_M": 2583, "VEH": 0.24}, 12: {"REV_M": 2500, "VEH": 0.33},
    13: {"REV_M": 2250, "VEH": 0.35}, 14: {"REV_M": 2583, "VEH": 0.36}, 15: {"REV_M": 2916, "VEH": 0.38},
    16: {"REV_M": 4583, "VEH": 0.45}, 17: {"REV_M": 3083, "VEH": 0.37}, 18: {"REV_M": 2083, "VEH": 0.20},
    19: {"REV_M": 1833, "VEH": 0.22}, 20: {"REV_M": 1916, "VEH": 0.23}
}

# --- 2. FONCTION API PARIS OPTIMISÉE ---
def obtenir_places_strict(type_voie, nom_rue):
    try:
        nom_pur = nom_rue.upper().strip()
        url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        
        # Requête plus précise : le nom doit contenir le mot et le type doit correspondre
        params = {"where": f"suggest(nomvoie, '{nom_pur}')", "limit": 100}
        regimes_valides = ["PAYANT ROTATIF", "PAYANT MIXTE", "GRATUIT"]
        
        res = requests.get(url, params=params, timeout=10).json()
        total = 0
        details_voies = set()
        
        if res and 'results' in res:
            for item in res['results']:
                nom_api = str(item.get('nomvoie', '')).upper()
                # On vérifie que le type (RUE, BD...) est bien présent dans le nom renvoyé par l'API
                if nom_pur in nom_api and type_voie.upper() in nom_api:
                    regime = str(item.get('regpri', '')).upper()
                    if any(r in regime for r in regimes_valides):
                        total += item.get('placal', 0)
                        details_voies.add(nom_api)
        
        return total, list(details_voies)
    except:
        return 15, ["Erreur API (Défaut 15)"]

# --- 3. FONCTIONS TECHNIQUES (MÉTÉO, SHEETS, ETC.) ---
def save_to_google_sheets(data_row):
    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds_dict = json.loads(st.secrets["gcp_service_account"]["json_data"])
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(credentials)
        sheet = client.open("SLOT_Beta1").sheet1
        sheet.append_row(data_row)
        return True
    except Exception as e:
        st.error(f"Erreur Sheets : {e}")
        return False

def get_live_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code"
        res = requests.get(url, timeout=5).json()
        temp = res['current']['temperature_2m']
        code = res['current']['weather_code']
        mto = "Beau" if code in [0, 1] else ("Nuageux" if code in [2, 3, 45, 48] else "Pluie")
        return mto, temp
    except: return "Beau", 18.0

@st.cache_resource
def load_assets():
    try: return joblib.load("modele_lightgbmDA.pkl"), joblib.load("preprocessorDA.pkl")
    except: return None, None

model, prepro = load_assets()

# --- 4. INTERFACE STREAMLIT ---
st.set_page_config(page_title="IA Parking Paris", page_icon="🅿️")
st.title("🅿️ Collecte Terrain Paris")

# Nouvelle ligne de saisie segmentée
c1, c2, c3 = st.columns([1, 1, 2])
num_voie = c1.text_input("N°")
type_voie = c2.selectbox("Type", ["Rue", "Boulevard", "Avenue", "Place", "Quai", "Impasse", "Allée"])
nom_rue = c3.text_input("Nom de la voie (ex: Rivoli)")

if st.button("🔍 ANALYSER LA ZONE"):
    if not nom_rue:
        st.warning("Entrez au moins le nom de la rue.")
    else:
        with st.spinner('Analyse en cours...'):
            adresse_complete = f"{num_voie} {type_voie} {nom_rue}, Paris"
            
            # A. Nombre de places RÉEL via Open Data avec le nouveau filtre
            nb_places_total, voies_trouvees = obtenir_places_strict(type_voie, nom_rue)
            
            # B. Géoloc pour Météo, Trafic et Arrondissement
            geo = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={adresse_complete}&limit=1").json()
            if geo['features']:
                feat = geo['features'][0]
                lat, lon = feat['geometry']['coordinates'][1], feat['geometry']['coordinates'][0]
                cp = feat['properties']['postcode']
                arrdt = int(cp[-2:])
                
                # C. Données dynamiques (Météo, Trafic, Socio-éco)
                socio = DATA_ARRDT.get(arrdt, {"REV_M": 2500, "VEH": 0.30})
                mto_live, temp_live = get_live_weather(lat, lon)
                
                score_trafic = 0
                API_MAPS = st.secrets.get("GOOGLE_API_KEY")
                if API_MAPS:
                    try:
                        t_url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={lat},{lon}&destinations={lat+0.001},{lon+0.001}&departure_time=now&key={API_MAPS}"
                        t_res = requests.get(t_url).json()
                        d_n = t_res['rows'][0]['elements'][0]['duration']['value']
                        d_t = t_res['rows'][0]['elements'][0]['duration_in_traffic']['value']
                        score_trafic = 100 if (d_t - d_n) > 120 else (50 if (d_t - d_n) > 60 else 0)
                    except: pass

                # D. Prédiction IA
                now = datetime.now(pytz.timezone('Europe/Paris'))
                X = pd.DataFrame([{
                    'RUE': nom_rue.upper(), 'VILLE': 'Paris', 'JOUR': now.strftime("%A"), 
                    'MTO': mto_live, 'TRAFIC': score_trafic, '% PARKING OC': 0.50, 
                    'NBR PLACES': nb_places_total, 'REVENUS / H': socio["REV_M"], 
                    'VEHICULES / H': socio["VEH"], 'TEMPERATURE': temp_live,
                    'HEURE_SIN': np.sin(2*np.pi*(now.hour*60+now.minute)/1440),
                    'HEURE_COS': np.cos(2*np.pi*(now.hour*60+now.minute)/1440)
                }])
                
                occ_pred = model.predict(prepro.transform(X))[0]
                pred_places = max(0, math.floor(nb_places_total * (1 - occ_pred)))

                # AFFICHAGE DES RÉSULTATS
                st.info(f"✅ **Open Data :** {nb_places_total} places trouvées sur : {', '.join(voies_trouvees)}")
                st.success(f"🤖 **IA :** {pred_places} places probablement libres actuellement.")
                
                st.session_state['releve'] = {
                    "Date": now.strftime("%d/%m/%Y"), "Heure": now.strftime("%H:%M"),
                    "Num": num_voie, "Rue": f"{type_voie} {nom_rue}", "Arrdt": arrdt,
                    "Pred_P": pred_places, "Pred_O": f"{round(occ_pred*100)}%",
                    "Mto": mto_live, "Temp": f"{temp_live}°C", "Trafic": score_trafic, "Total_API": nb_places_total
                }
            else:
                st.error("Localisation impossible. Vérifiez le nom de la rue.")

# --- 5. ENREGISTREMENT ---
if 'releve' in st.session_state:
    st.divider()
    reel = st.number_input("Places RÉELLES observées par Louis :", min_value=0, step=1)
    if st.button("💾 SAUVEGARDER LE RELEVÉ"):
        d = st.session_state['releve']
        ligne = [d["Date"], d["Heure"], d["Num"], d["Rue"], d["Arrdt"], d["Pred_P"], d["Pred_O"], reel, d["Mto"], d["Temp"], d["Trafic"], d["Total_API"]]
        if save_to_google_sheets(ligne):
            st.balloons()
            st.success("Données envoyées !")
            del st.session_state['releve']
