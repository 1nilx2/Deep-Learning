"""Conservative, corpus-specific Katie smoke-test preparation (not a general CHAT parser)."""
import hashlib
import json
import re
import wave
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/raw/gleason'
TIME = re.compile(r'\x15(\d+)_(\d+)\x15')


def candidates(text):
    records = []
    current = None
    corrupted = False
    for number, line in enumerate(text.splitlines(), 1):
        if line.startswith('@Comment:') and 'Starting at this point' in line:
            corrupted = True
        if line.startswith('*'):
            speaker, content = line[1:].split(':', 1)
            current = dict(line=number, speaker=speaker, raw=content.strip(), corrupted=corrupted)
            records.append(current)
        elif line.startswith(('\t', ' ')) and current is not None:
            current['raw'] += ' ' + line.strip()
        else:
            current = None
    for record in records:
        stamps = TIME.findall(record['raw'])
        record['span'] = tuple(map(int, stamps[0])) if len(stamps) == 1 else None
    return records


def reject_reason(row, records, duration_ms):
    if row['corrupted']:
        return 'after_explicit_corruption_warning'
    if row['span'] is None:
        return 'missing_or_multiple_timestamps'
    start, end = row['span']
    if not 0 <= start < end <= duration_ms:
        return 'invalid_audio_bounds'
    if not 700 <= end - start <= 10000:
        return 'duration_outside_0.7_to_10_seconds'
    content = TIME.sub('', row['raw']).strip()
    # Do not guess how retracing, substitutions, omissions or events were realized.
    if re.search(r'[\[\]<>@&+=/\\\d]|\b(?:xxx|yyy|www)\b', content, re.I):
        return 'complex_chat_or_unintelligible'
    if any(other['speaker'] != 'CHI' and other['span'] and
           max(start, other['span'][0]) < min(end, other['span'][1]) for other in records):
        return 'timestamp_overlap_with_other_speaker'
    return None


def clean_text(raw):
    text = TIME.sub('', raw)
    text = re.sub(r'\((?:\.+|\d*\.?\d+)\)', ' ', text)  # pause annotations only
    text = text.replace('_', ' ').replace(':', '')  # compounds and lengthening
    text = re.sub(r'\s+', ' ', text).strip()
    if re.search(r'[^a-zA-Z\s\x27.,?!-]', text):
        return None
    if len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)) < 2:
        return None
    return text


def main():
    transcript = SOURCE / 'mother-katie.cha'
    audio = SOURCE / 'Mother/katie.wav'
    records = candidates(transcript.read_text())
    clips = ROOT / 'data/clips/gleason-katie'
    clips.mkdir(parents=True, exist_ok=True)
    accepted, rejected, seen = [], [], set()
    with wave.open(str(audio)) as source:
        rate = source.getframerate()
        duration = source.getnframes() / rate * 1000
        for row in records:
            if row['speaker'] != 'CHI':
                continue
            reason = reject_reason(row, records, duration)
            text = clean_text(row['raw']) if reason is None else None
            if reason is None and text is None:
                reason = 'unsupported_text_or_less_than_two_words'
            key = re.sub(r'[^a-z\x27 ]', '', text.lower()).strip() if text else None
            if reason is None and key in seen:
                reason = 'duplicate_normalized_transcript'
            if reason:
                rejected.append(dict(line=row['line'], reason=reason))
                continue
            seen.add(key)
            start, end = row['span']
            first, last = round(start * rate / 1000), round(end * rate / 1000)
            source.setpos(first)
            frames = source.readframes(last - first)
            if len(frames) != (last - first) * source.getsampwidth() * source.getnchannels():
                raise ValueError('Truncated source audio')
            ident = f'gleason-katie-mother-L{row["line"]:04}'
            path = clips / f'{ident}.wav'
            with wave.open(str(path), 'wb') as target:
                target.setparams(source.getparams())
                target.writeframes(frames)
            accepted.append(dict(id=ident, child_id='gleason-katie', session_id='gleason-katie-mother',
                                 text=text, audio_path=str(path.relative_to(ROOT)), age_months=38,
                                 age_chat='3;02.02', child_group_original='TD', diagnosis=None,
                                 source_start_ms=start, source_end_ms=end, source_line=row['line'],
                                 transcript_raw=row['raw'], manual_alignment_verified=False))
    if len(accepted) < 2:
        raise ValueError('Too few usable clips')
    output = ROOT / 'data/katie_smoke.jsonl'
    output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in accepted))
    summary = dict(scope='Single-child technical smoke test; not multi-child research pilot',
                   child_utterances=sum(r['speaker']=='CHI' for r in records), accepted=len(accepted),
                   rejection_counts=dict(Counter(r['reason'] for r in rejected)), rejected=rejected,
                   source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (audio, transcript)},
                   source_audio_url='https://media.talkbank.org/childes/Eng-NA/Gleason/Mother/katie.wav?f=save',
                   source_transcripts_url='https://talkbank.org/data/childes/Eng-NA/Gleason?f=zip',
                   audio_sample_rate=rate, audio_duration_seconds=duration/1000,
                   manual_listening_completed=False,
                   notes=['No embedding scores used in selection.',
                          'Age in completed months; original CHAT age retained.',
                          'Topic labels omitted: toyplay is a session activity, not utterance topic.',
                          'No audio past explicit corruption warning is included.',
                          'Unannotated overlap and timestamp quality still require listening.'])
    (ROOT / 'data/preparation_report.json').write_text(json.dumps(summary, indent=2))
    review = ['# Katie clip listening checklist', '',
              'Not yet manually verified. Mark each item after checking child speaker, transcript and boundaries.', '']
    review += [f'- [ ] {r["id"]}: {r["text"]} ([audio]({r["audio_path"].removeprefix("data/")}))' for r in accepted]
    (ROOT / 'data/LISTENING_REVIEW.md').write_text('\n'.join(review)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rejected','source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
