"""Offline Poison-Hunter label integrity and 200-wallet manifest audit."""
import csv
import json
from pathlib import Path
import re
import zipfile
import sys
if __package__:
    from .run import atomic_json,read_manifest
else:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from external_evaluation.run import atomic_json,read_manifest
HERE=Path(__file__).resolve().parent

def main():
    with zipfile.ZipFile(HERE/'Poison-Hunter-main.zip') as archive:
        names=archive.namelist()
        def get(filename):
            matches=[n for n in names if n.endswith('/'+filename)]
            if len(matches)!=1:raise ValueError('Could not resolve label file: '+filename)
            raw=archive.read(matches[0]).decode()
            result={x.strip().lower() for x in raw.splitlines() if x.strip()}
            if not all(re.fullmatch('0x[0-9a-f]{40}',x) for x in result):raise ValueError('Bad address in source labels')
            return result
        phishing=get('phishing_address_groundtruth.txt')
        benign=get('benign_addresses_on_etherscan.txt')
        detected=get('phishing_address_we_detected_TP.txt')
    rows=read_manifest(HERE/'manifest.csv')
    invalid=[r for r in rows if r['address'] not in (phishing if r['label']=='1' else benign)]
    if invalid or phishing & benign or not detected <= phishing:raise ValueError('Label mismatch or source integrity problem')
    d2_file=HERE/'D2_training_addresses.txt'
    d2={s.strip().lower() for s in d2_file.read_text().splitlines() if s.strip()} if d2_file.is_file() else set()
    d2_overlap=sum(r['address'] in d2 for r in rows)
    if d2_overlap:raise ValueError(f'{d2_overlap} manifest wallets appear in D2 training data')
    report={'known_D2_training_addresses':len(d2),'manifest_D2_overlap':d2_overlap,
            'source_pool_D2_overlap':len((phishing|benign)&d2),
            'poison_hunter_phishing_addresses':len(phishing),'poison_hunter_benign_addresses':len(benign),
            'poison_hunter_groundtruth_total':len(phishing|benign),
            'detected_TP_subset_count_not_independent':len(detected),
            'manifest_total':len(rows),'manifest_phishing':sum(r['label']=='1' for r in rows),
            'manifest_benign':sum(r['label']=='0' for r in rows),
            'duplicate_or_mislabeled_addresses':0,'labels_consistent':True,
            'possible_D1_overlap':'unknown: D1 lacks address IDs',
            'dataset_is_wallet_feature_matrix':False,
            'model_predictions_from_labels_alone_possible':False}
    if len(rows)!=200 or report['manifest_phishing']!=100 or report['manifest_benign']!=100:
        raise ValueError('Manifest was expected to contain 100+100')
    out=HERE/'results';out.mkdir(exist_ok=True)
    atomic_json(out/'dataset_audit.json',report)
    print(json.dumps(report,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
