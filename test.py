# test_v2.py — à lancer depuis la racine du repo
import numpy as np
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'experiments/togo-fiscal')
from core import PopulationRSI, EntityScorer
from rules import RULE_IDS, make_priors, best_f1, extract_signals
import pandas as pd

np.random.seed(42)
priors = make_priors()

smart_penalty = {
    'R1_TVA': 0.7, 'R2_IS': 0.6, 'R3_IMF': 0.5,
    'R4_TPU': 0.7, 'R5_IRPP': 0.6, 'R6_PAT': 0.7,
    'R7_DECL': 0.6, 'R8_BANK': 0.7,
}

for name, path in [('MCAR', 'datasets/togo-fiscal/dataset-v2-mcar.csv'),
                    ('MNAR', 'datasets/togo-fiscal/dataset-v2-mnar.csv')]:
    df = pd.read_csv(path)
    data = extract_signals(df, period=1)
    eids = data['entity_ids']
    
    for pen_name, pen in [('neutral', None), ('smart', smart_penalty)]:
        scorer = EntityScorer(RULE_IDS, silence_penalty=pen)
        ent = scorer.score(data['sigs_raw'], data['apps'], eids)
        
        f1s = []
        for rid in RULE_IDS:
            ae = [e for e in eids if data['apps'][rid].get(e, False)]
            yt = np.array([data['labels_rule'][rid].get(e, 0) for e in ae])
            if len(np.unique(yt)) < 2: continue
            ys = np.array([ent['entity_rules'].get((e, rid), {}).get('nc_score', 0.5) for e in ae])
            f1, _ = best_f1(yt, ys)
            f1s.append(f1)
        
        print(f"{name} | {pen_name:>8} | Mean F1 = {np.mean(f1s):.3f}")
```

Tu devrais obtenir :
```
MCAR |  neutral | Mean F1 = 0.741
MCAR |    smart | Mean F1 = 0.733
MNAR |  neutral | Mean F1 = 0.775
MNAR |    smart | Mean F1 = 0.779