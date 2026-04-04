import streamlit as st
import requests, pytz, joblib, math, numpy as np, pandas as pd, re
from datetime import datetime

# --- 1. DÉFINITION DES FONCTIONS ---
def extraire_nom_propre(adresse):
    """Nettoyage pour correspondre aux noms de rues de l'IA"""
    s = adresse.upper()
    s = re.sub(r'\d+', '', s) # Enlève les numéros (ex: 11 -> "")
    mots_a_supprimer = ["RUE DU ", "RUE DES ", "RUE DE LA ", "RUE DE ", "RUE ", "AVENUE ", "BOULEVARD ", "PLACE ", "D'"]
    for mot in mots_a_supprimer:
        s = s.replace(mot, "")
    return s.strip()

# --- 2. CONFIGURATION DE L'INTERFACE ---
st.set_page_config(page_title="IA Parking Paris", page_icon="🅿️")

# Style mobile (bouton large)
st.markdown("""
    <style>
    .stButton>button { width: 100%; height: 3em; background-color: #007bff; color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🅿️ IA Parking Paris")
st.caption("Aide au stationnement en temps réel - Version V54")

# Données socio-éco par arrondissement
DATA_ARRDT = {
    1: {"REV_M": 2916, "VEH": 0.32}, 2: {"REV_M": 2666, "VEH": 0.28}, 3: {"REV_M": 2833, "VEH": 0.25},
    4: {"REV_M": 2750, "VEH": 0.26}, 5: {"REV_M": 3166, "VEH": 0.35}, 6: {"REV_M": 3750, "VEH": 0.38},
    7: {"REV_M": 4166, "VEH": 0.42}, 8: {"REV_M": 4000, "VEH": 0.40}, 9: {"REV_M": 3000, "VEH": 0.30},
    10: {"REV_M": 2333, "VEH": 0.22}, 11: {"REV_M": 2583, "VEH": 0.24}, 12: {"REV_M": 2500, "VEH": 0.33},
    13: {"REV_M": 2250, "VEH": 0.35}, 14: {"REV_M": 2583, "VEH": 0.36}, 15: {"REV_M": 2916, "VEH": 0.38},
    16: {"REV_M": 4583, "VEH": 0.45}, 17: {"REV_M": 3083, "VEH": 0.37}, 18: {"REV_M": 2083, "VEH": 0.20},
    19: {"REV_M": 1833, "VEH": 0.22}, 20: {"REV_M": 1916, "VEH": 0.23}
}

# --- 3. CHARGEMENT DES MODÈLES ---
@st.cache_resource
def load_assets():
    try:
        m = joblib.load("modele_lightgbmDA.pkl")
        p = joblib.load("preprocessorDA.pkl")
        return m, p
    except:
        return None, None

model, prepro = load_assets()
API_KEY_GOOGLE = st.secrets.get("GOOGLE_API_KEY")

# --- 4. FORMULAIRE UTILISATEUR ---
adresse_user = st.text_input("📍 Saisissez l'adresse à Paris :", placeholder="ex: 11 rue du commandant lamy")

if st.button("ANALYSER LA ZONE"):
    if not adresse_user:
        st.warning("Veuillez entrer une adresse.")
    elif model is None:
        st.error("Modèle IA introuvable.")
    else:
        with st.spinner('Analyse IA en cours...'):
            try:
                # 1. Nettoyage du nom de rue
                nom_pur = extraire_nom_propre(adresse_user)
                
                # 2. API Paris (Comptage des places théoriques)
                url_p = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
                params_p = {"where": f"suggest(nomvoie, '{nom_pur}')", "limit": 100}
                data_p = requests.get(url_p, params=params_p).json()
                
                nb_places = sum(item.get('placal', 0) for item in data_p.get('results', []))
                if nb_places == 0: nb_places = 5

                # 3. Géolocalisation (Lat/Lon) et Arrondissement
                geo = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={adresse_user.replace(' ', '+')}+Paris&limit=1").json()
                feat = geo['features'][0]
                lat, lon = feat['geometry']['coordinates'][1], feat['geometry']['coordinates'][0]
                arrdt = int(feat['properties']['postcode'][-2:])

                # 4. Trafic temps réel (Google)
                score_trafic_pct = 0
                if API_KEY_GOOGLE:
                    t_url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={lat},{lon}&destinations={lat+0.002},{lon+0.002}&departure_time=now&key={API_KEY_GOOGLE}"
                    t_data = requests.get(t_url).json()
                    if 'rows' in t_data:
                        elem = t_data['rows'][0]['elements'][0]
                        if 'duration_in_traffic' in elem:
                            retard = elem['duration_in_traffic']['value'] - elem['duration']['value']
                            score_trafic_pct = 100 if retard > 120 else (50 if retard > 60 else 0)

                # 5. Prédiction IA
                now = datetime.now(pytz.timezone('Europe/Paris'))
                stats = DATA_ARRDT.get(arrdt, {"REV_M": 2500, "VEH": 0.30})
                
                X = pd.DataFrame([{
                    'RUE': nom_pur, 'VILLE': 'Paris', 'JOUR': now.strftime("%A"), 'MTO': "Beau",
                    'TRAFIC': score_trafic_pct, '% PARKING OC': 0.50, 'NBR PLACES': nb_places,
                    'REVENUS / H': stats['REV_M'], 'VEHICULES / H': stats['VEH'],
                    'TEMPERATURE': 18.0,
                    'HEURE_SIN': np.sin(2*np.pi*(now.hour*60+now.minute)/1440),
                    'HEURE_COS': np.cos(2*np.pi*(now.hour*60+now.minute)/1440)
                }])

                occ_pred = model.predict(prepro.transform(X))[0]
                libres = max(0, math.floor(nb_places * (1 - occ_pred)))

                # 6. Affichage des résultats
                st.divider()
                st.subheader(f"Résultat pour : {nom_pur}")
                
                col1, col2 = st.columns(2)
                col1.metric("Places Libres (IA)", f"{libres}")
                col2.metric("Taux d'occupation", f"{round(occ_pred * 100)}%")

                if libres > 3:
                    st.success("🟢 VERT : Stationnement facile !")
                elif libres > 0:
                    st.warning("🟠 ORANGE : Zone tendue.")
                else:
                    st.error("🔴 ROUGE : Zone saturée.")

                st.caption(f"Mis à jour à {now.strftime('%H:%M')} | Arrondissement : {arrdt}")

            except Exception as ex:
                st.error(f"Erreur d'analyse : {ex}")
