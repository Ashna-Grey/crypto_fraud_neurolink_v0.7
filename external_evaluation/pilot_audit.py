"""Parse archived five-wallet pilot accurately. Does not fabricate missing scores."""
import csv
import json
import sys
from pathlib import Path
if __package__:
    from .metrics import summarize
    from .run import atomic_json
else:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from external_evaluation.metrics import summarize
    from external_evaluation.run import atomic_json
HERE=Path(__file__).resolve().parent

def main():
    rows=[]
    for raw in csv.DictReader((HERE/'baseline_pilot'/'predictions.csv').open(newline='')):
        decision=raw.get('prediction')
        status=('abstained_manual_review' if decision=='MANUAL_REVIEW' else 'error' if raw.get('status')=='error' else 'other')
        rows.append({'address':raw['address'],'label':int(raw['label']),
                     'complete':raw.get('complete')=='True','status':status,
                     'decision':decision,'probability':raw.get('probability')})
    report=summarize(rows,5,provenance='archived_five_wallet_poison_hunter_pilot')
    report['note']='Historical pilot output imported from user files; not a new RPC/API run.'
    report['manual_review_is_not_correct_prediction']=True
    out=HERE/'results';out.mkdir(exist_ok=True)
    atomic_json(out/'pilot_audit.json',report)
    print(json.dumps(report,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
