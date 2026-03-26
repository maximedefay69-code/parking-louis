import streamlit as st
import requests, pytz, joblib, math, numpy as np, pandas as pd, re
from datetime import datetime

# Données socio-éco V54
DATA_ARRDT = {
    1: {"REV_M": 2916, "VEH": 0.32}, 2: {"REV_M": 2666, "VEH": 0.28}, 3: {"REV_M": 2833, "VEH": 0.25},
    4: {"REV_M": 2750, "VEH": 0.26}, 5: {"REV_M": 3166, "VEH": 0.35}, 6: {"REV_M": 3750, "VEH": 0.38},
    7: {"REV_M": 4166, "VEH": 0.42}, 8: {"REV_M": 4000, "VEH": 0.40}, 9: {"REV_M": 3000, "VEH": 0.30},
    10: {"REV_M": 2333, "VEH": 0.22}, 11: {"REV_M": 2583, "VEH": 0.24}, 12: {"REV_M": 2500, "VEH": 0.33},
    13: {"REV_M": 2250, "VEH": 0.35}, 14: {"REV_M": 2583, "VEH": 0.36}, 15: {"REV_M": 2916, "VEH": 0.38},
    16: {"REV_M": 4583, "VEH": 0.45}, 17: {"REV_M": 3083, "VEH": 0.37}, 18: {"REV_M": 2083, "VEH": 0.20},
    19: {"REV_M": 1833, "VEH": 0.22}, 20: {"REV_M": 1916, "VEH": 0.23}
}

st.set_page_config(page_title="Parking Paris Live", page_icon="🅿️")
st.title("🅿️ IA Parking Paris")

# Récupération sécurisée de la clé
API_KEY_GOOGLE = st.secrets["GOOGLE_API_KEY"]

@st.cache_resource
def load_assets():
    m = joblib.load("modele_lightgbmDA.pkl")
    p = joblib.load("preprocessorDA.pkl")
    return m, p

model, prepro = load_assets()

adresse = st.text_input("📍 Adresse (ex: 11 rue du commandant lamy) :")

if st.button("LANCER L'ANALYSE", use_container_width=True):
    if adresse:
        with st.spinner('Analyse...'):
            # Nettoyage
            nom_pur = re.sub(r'\d+', '', adresse.upper()).replace("RUE DU ","").replace("RUE DE ","").replace("RUE ","").strip()
            
            # API Paris
            res_p = requests.get("https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records", 
                                 params={"where": f"suggest(nomvoie, '{nom_pur}')", "limit": 100}).json()
            nb_places = sum(item.get('placal', 0) for item in res_p.get('results', []) if any(r in str(item.get('regpri','')).upper() for r in ["PAYANT", "GRATUIT"])) or 5

            # GPS & Trafic
            geo = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={adresse.replace(' ', '+')}+Paris&limit=1").json()
            lat, lon = geo['features'][0]['geometry']['coordinates'][1], geo['features'][0]['geometry']['coordinates'][0]
            arrdt = int(geo['features'][0]['properties']['postcode'][-2:])
            
            t_data = requests.get(f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={lat},{lon}&destinations={lat+0.002},{lon+0.002}&departure_time=now&key={API_KEY_GOOGLE}").json()
            retard = t_data['rows'][0]['elements'][0]['duration_in_traffic']['value'] - t_data['rows'][0]['elements'][0]['duration']['value'] if 'duration_in_traffic' in t_data['rows'][0]['elements'][0] else 0
            score_t = 0 if retard < 60 else (50 if retard <= 120 else 100)

            # IA
            now = datetime.now(pytz.timezone('Europe/Paris'))
            stats = DATA_ARRDT.get(arrdt, {"REV_M": 2500, "VEH": 0.30})
            X = pd.DataFrame([{'RUE': nom_pur, 'VILLE': 'Paris', 'JOUR': now.strftime("%A"), 'MTO': "Beau", 'TRAFIC': score_t, 'NBR PLACES': nb_places, 'REVENUS / H': stats['REV_M'], 'VEHICULES / H': stats['VEH'], 'TEMPERATURE': 18.0, 'HEURE_SIN': np.sin(2*np.pi*(now.hour*60+now.minute)/1440), 'HEURE_COS': np.cos(2*np.pi*(now.hour*60+now.minute)/1440)}])
            
            occ = model.predict(prepro.transform(X))[0]
            libres = max(0, math.floor(nb_places * (1 - occ)))

            # Affichage
            st.divider()
            st.metric("Places Libres Estimées", f"{libres} / {nb_places}")
            if libres > 2: st.success("🟢 ZONE FACILE")
            else: st.error("🔴 ZONE SATURÉE")
