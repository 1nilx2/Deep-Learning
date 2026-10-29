"""Frozen sample selection for expanded Exp 1; alignment is assumed at user's request."""
import hashlib
import json
import random
import re
import wave
import zipfile
from collections import Counter
from prepare_gleason import ROOT, SOURCE, candidates
from prepare_dialogues import windows

SESSIONS = [('Father','andy'),('Father','frank'),('Father','olivia'),
            ('Mother','andy'),('Mother','charlie'),('Mother','frank'),
            ('Mother','isadora'),('Mother','katie')]
SEED = 20261028
CAP = 30


def main():
    rng = random.Random(SEED)
    destination = ROOT/'data/exp1_full'
    destination.mkdir(parents=True,exist_ok=True)
    rows, audit = [], []
    prior = [json.loads(l) for l in (ROOT/'data/katie_dialogues_verified3.jsonl').read_text().splitlines()]
    verified_spans = {(r['source_start_ms'],r['source_end_ms']) for r in prior}
    with zipfile.ZipFile(SOURCE/'Gleason.zip') as archive:
        for parent, child in SESSIONS:
            key = f'{parent}/{child}.cha'
            transcript = archive.read(key).decode('utf-8-sig')
            (destination/f'{parent}-{child}.cha').write_text(transcript)
            audio = SOURCE/parent/f'{child}.wav'
            with wave.open(str(audio)) as source:
                rate, width, channels = source.getframerate(),source.getsampwidth(),source.getnchannels()
                payload = source.readframes(source.getnframes())
                actual_frames = len(payload)//(width*channels)
                groups = windows(candidates(transcript),actual_frames/rate*1000)
                selected_indices = sorted(rng.sample(range(len(groups)), min(CAP,len(groups))))
                audit.append(dict(session=f'{parent}-{child}', eligible_windows=len(groups), selected_before_duration_qc=len(selected_indices), selected=0, duration_qc_exclusions=[],
                                  actual_seconds=actual_frames/rate, declared_frames=source.getnframes(),
                                  truncated=actual_frames<source.getnframes(),
                                  audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
                                  transcript_sha256=hashlib.sha256(archive.read(key)).hexdigest(),
                                  source_url=f'https://media.talkbank.org/childes/Eng-NA/Gleason/{parent}/{child}.wav?f=save'))
                idline=next(l for l in transcript.splitlines() if l.startswith('@ID:') and '|CHI|' in l)
                age=idline.split('|')[3]
                for j in selected_indices:
                    turns=groups[j]
                    start,end=turns[0]['span'][0],turns[-1]['span'][1]
                    first,last=round(start*rate/1000),round(end*rate/1000)
                    if not 10 <= (last-first)/rate <= 30:
                        audit[-1]['duration_qc_exclusions'].append({'source_start_ms':start,'source_end_ms':end,'reason':'outside_10_to_30_seconds'})
                        continue
                    audit[-1]['selected'] += 1
                    ident=f'gleason-{parent.lower()}-{child}-{start:07d}-{end:07d}'
                    path=destination/f'{ident}.wav'
                    clip=payload[first*width*channels:last*width*channels]
                    assert len(clip)==(last-first)*width*channels
                    with wave.open(str(path),'wb') as target:
                        target.setparams(source.getparams())
                        target.writeframes(clip)
                    verified=parent=='Mother' and child=='katie' and (start,end) in verified_spans
                    rows.append(dict(id=ident, child_id=f'gleason-{child}',session_id=f'gleason-{parent.lower()}-{child}',
                                     text='\n'.join(t['spoken'] for t in turns),audio_path=str(path.relative_to(ROOT)),
                                     source_start_ms=start,source_end_ms=end,first_frame=first,last_frame_exclusive=last,
                                     sample_rate=rate,age_chat=age,analysis_unit='all_speaker_dialogue',
                                     manual_alignment_verified=verified,alignment_assumed_by_user=not verified,
                                     verification_status='user_reviewed' if verified else 'assumed_aligned_at_user_request',
                                     verification_source='User reviewed first three Katie dialogues' if verified else
                                     'User authorized assuming alignment for expanded evaluation',
                                     topic=None,topic_status='unannotated',
                                     turns=[dict(speaker=t['speaker'],text=t['spoken'],source_line=t['line'],
                                                 source_start_ms=t['span'][0],source_end_ms=t['span'][1]) for t in turns]))
    (ROOT/'data/exp1_full_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary=dict(seed=SEED,selection='Fixed-seed random sample of at most 30 eligible nonoverlapping windows per session',
                 target_seconds=20,minimum_seconds=10,maximum_seconds=30,max_between_turn_gap_ms=300,
                 n_pairs=len(rows),n_children=len({r['child_id'] for r in rows}),n_sessions=len(audit),
                 verified_pairs=sum(r['manual_alignment_verified'] for r in rows),
                 assumed_pairs=sum(r['alignment_assumed_by_user'] for r in rows),
                 per_child=dict(Counter(r['child_id'] for r in rows)),sessions=audit,
                 manifest_sha256=hashlib.sha256((ROOT/'data/exp1_full_manifest.jsonl').read_bytes()).hexdigest(),
                 total_seconds=sum((r['last_frame_exclusive']-r['first_frame'])/r['sample_rate'] for r in rows))
    (ROOT/'data/exp1_full_preparation.json').write_text(json.dumps(summary,indent=2))
    for session in {r['session_id'] for r in rows}:
        spans=sorted((r['source_start_ms'],r['source_end_ms']) for r in rows if r['session_id']==session)
        assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
    print(json.dumps({k:v for k,v in summary.items() if k!='sessions'},indent=2))


if __name__=='__main__':
    main()
