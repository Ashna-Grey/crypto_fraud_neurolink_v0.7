"""Optional: migrate old five-wallet pilot's cached Alchemy responses without recollection.

Usage: python -m external_evaluation.import_previous_cache --from-dir 'C:\\...\\Testing\\results\\cache'
Only import complete, snapshot-matching wallet histories. No API keys are transferred.
"""
import argparse
import json
from pathlib import Path
import sys
if __package__:
    from .run import atomic_json,read_manifest
else:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from external_evaluation.run import atomic_json,read_manifest
HERE=Path(__file__).resolve().parent

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--from-dir',type=Path,required=True,help='Old Testing/results/cache folder')
    a.add_argument('--outdir',type=Path,default=HERE/'results')
    args=a.parse_args()
    if not args.from_dir.is_dir():raise SystemExit('Previous cache directory does not exist')
    allowed={r['address'] for r in read_manifest(HERE/'manifest.csv')}
    block_file=args.outdir/'snapshot_block.txt'
    snapshot=block_file.read_text().strip() if block_file.exists() else (HERE/'reference_snapshot.txt').read_text().strip()
    cache=args.outdir/'cache';cache.mkdir(parents=True,exist_ok=True)
    imported=0;skipped=0
    for file in sorted(args.from_dir.glob('0x*.json')):
        try:
            record=json.loads(file.read_text(encoding='utf-8'))
            address=str(record['address']).lower()
            if (address not in allowed or str(record.get('snapshot','')).lower()!=snapshot.lower()
                or record.get('incoming_complete') is not True
                or record.get('outgoing_complete') is not True
                or not isinstance(record.get('incoming'),list)
                or not isinstance(record.get('outgoing'),list)
                or not isinstance(record.get('balance'),(int,float))):
                skipped+=1;continue
            for direction in ('incoming','outgoing'):
                target=cache/f'{address}_{direction}.json'
                if not target.exists():
                    atomic_json(target,{'address':address,'direction':direction,'snapshot':snapshot,
                                        'page_key':None,'pages':1,'complete':True,'transfers':record[direction]})
            balance_file=cache/f'{address}_balance.json'
            if not balance_file.exists():
                atomic_json(balance_file,{'snapshot':snapshot,'address':address,'balance':record['balance']})
            imported+=1
        except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError):skipped+=1
    if imported and not block_file.exists():block_file.write_text(snapshot,encoding='utf-8')
    print(f'Complete wallets imported: {imported}; skipped: {skipped}')
    return 0

if __name__=='__main__':raise SystemExit(main())
