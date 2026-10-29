"""Subset already extracted vectors to the duration-QC manifest, without recomputing embeddings."""
import hashlib
import json
import shutil
import numpy as np
from experiment import ROOT


def main():
    raw=ROOT/'outputs/exp1_full_before_qc'
    final=ROOT/'outputs/exp1_full'
    if not raw.exists():
        shutil.move(str(final),str(raw))
    old=json.loads((raw/'rows.json').read_text())
    raw_manifest=''.join(json.dumps(r)+'\n' for r in old)
    rawpath=ROOT/'data/exp1_full_before_qc_manifest.jsonl'
    rawpath.write_text(raw_manifest)
    provenance=json.loads((raw/'provenance.json').read_text())
    provenance['manifest_sha256']=hashlib.sha256(rawpath.read_bytes()).hexdigest()
    provenance['manifest_note']='Reconstructed exact JSONL from extraction rows after duration QC was applied during inference.'
    (raw/'provenance.json').write_text(json.dumps(provenance,indent=2))
    path=ROOT/'data/exp1_full_manifest.jsonl'
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    lookup={r['id']:i for i,r in enumerate(old)}
    indices=[lookup[r['id']] for r in rows]
    for r,i in zip(rows,indices):
        assert r==old[i], 'Metadata changed beyond row exclusion'
    final.mkdir(exist_ok=True)
    with np.load(raw/'embeddings.npz') as d:
        np.savez_compressed(final/'embeddings.npz',text=d['text'][indices],audio=d['audio'][indices])
    (final/'rows.json').write_text(json.dumps(rows,indent=2))
    provenance.update(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      manifest_note='Final duration-QC subset; model vectors reused by exact row ID.',
                      parent_extraction='outputs/exp1_full_before_qc', excluded_ids=[r['id'] for r in old if r['id'] not in {x['id'] for x in rows}])
    provenance['audio_sha256']={r['id']:provenance['audio_sha256'][r['id']] for r in rows}
    (final/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('Final duration-QC vectors:',len(rows))


if __name__=='__main__':
    main()
