import sys
sys.path.insert(0, "src")
from rsi_engine import RSIEngine

engine = RSIEngine.for_togo(period="2022_2024")

obs = {
    "obs_ca_declare": 72_000_000,
    "obs_tva_declaree": 0,
    "obs_tva_missing": False,
    "obs_tva_assujetti_declare": False,
    "obs_retard_paiement_jours": 45,
    "obs_has_compte_bancaire": True,
    "obs_utilise_facturation_electronique": False,
    "obs_is_declare": 0,
    "obs_is_missing": True,
    "obs_benefice_declare": 0,
    "obs_benefice_missing": True,
    "obs_ratio_sous_declaration": 0.75,
}

result = engine.predict_compliance(obs)
print(result)