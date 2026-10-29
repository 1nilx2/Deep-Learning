"""Prepare whole-turn, all-speaker dialogue candidates; never certify by timestamps alone."""
import json
import re
import wave
from pathlib import Path
from prepare_gleason import ROOT, SOURCE, TIME, candidates


def spoken_text(raw):
    text = TIME.sub('', raw)
    text = re.sub(r'\((?:\.+|\d*\.?\d+)\)', ' ', text)
    text = text.replace('_', ' ').replace(':', '')
    if re.search(r"[^A-Za-z\s'.,?!-]", text) or re.search(r'\b(?:xxx|yyy|www)\b', text, re.I):
        return None
    return re.sub(r'\s+', ' ', text).strip() or None


def windows(records, duration_ms, target_ms=20000):
    runs, run = [], []
    for row in records:
        span = row['span']
        text = spoken_text(row['raw'])
        good = not row['corrupted'] and span and 0 <= span[0] < span[1] <= duration_ms and text
        if good:
            good = not any(other is not row and other['span'] and
                           max(span[0], other['span'][0]) < min(span[1], other['span'][1])
                           for other in records)
        contiguous = good and (not run or 0 <= span[0] - run[-1]['span'][1] <= 300)
        if not contiguous:
            if run:
                runs.append(run)
            run = []
        if good:
            run.append(dict(row, spoken=text))
    if run:
        runs.append(run)
    result = []
    for run in runs:
        current = []
        for row in run:
            if current and row['span'][1] - current[0]['span'][0] > 30000:
                if current[-1]['span'][1] - current[0]['span'][0] >= 10000:
                    result.append(current)
                current = []
            current.append(row)
            if current[-1]['span'][1] - current[0]['span'][0] >= target_ms:
                result.append(current)
                current = []
        if current and current[-1]['span'][1] - current[0]['span'][0] >= 10000:
            result.append(current)
    return result


def main():
    audio = SOURCE / 'Mother/katie.wav'
    records = candidates((SOURCE / 'mother-katie.cha').read_text())
    output = ROOT / 'data/dialogues'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with wave.open(str(audio)) as source:
        rate = source.getframerate()
        groups = windows(records, source.getnframes() / rate * 1000)
        for i, turns in enumerate(groups, 1):
            start, end = turns[0]['span'][0], turns[-1]['span'][1]
            first, last = round(start * rate / 1000), round(end * rate / 1000)
            source.setpos(first)
            frames = source.readframes(last - first)
            if len(frames) != (last-first)*source.getsampwidth()*source.getnchannels():
                raise ValueError('Incomplete audio')
            ident = f'katie-dialogue-{i:02}'
            path = output / f'{ident}.wav'
            with wave.open(str(path), 'wb') as target:
                target.setparams(source.getparams())
                target.writeframes(frames)
            rows.append(dict(id=ident, child_id='gleason-katie', session_id='gleason-katie-mother',
                             text='\n'.join(t['spoken'] for t in turns),
                             audio_path=str(path.relative_to(ROOT)),
                             source_start_ms=start, source_end_ms=end, first_frame=first, last_frame_exclusive=last,
                             sample_rate=rate, analysis_unit='all_speaker_dialogue',
                             turns=[dict(speaker=t['speaker'], text=t['spoken'], source_line=t['line'],
                                         source_start_ms=t['span'][0], source_end_ms=t['span'][1], raw=t['raw']) for t in turns],
                             manual_alignment_verified=False, verification_status='pending_content_and_boundary_review'))
    (ROOT/'data/katie_dialogues_candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    review = ['# 긴 대화 구간 검증 대기', '',
              '전사 타임스탬프에 따른 후보입니다. 실제 음성의 내용·화자·시작·끝 일치는 아직 검증되지 않았습니다.',
              '모든 화자의 말을 빠짐없이 포함했는지, 첫·마지막 발화가 잘리지 않았는지 확인해야 합니다.',
              '자동 텍스트 정제: pause/장음 표기 제거, 밑줄 단어 분리. 화자 이름은 모델 입력에서 제외하고 아래 검토본에만 표시합니다.', '']
    for r in rows:
        review += [f'## {r["id"]}', '', f'원본 {r["source_start_ms"]/1000:.3f}–{r["source_end_ms"]/1000:.3f}초; {(r["last_frame_exclusive"]-r["first_frame"])/rate:.3f}초', '',
                   f'![audio]({(ROOT/r["audio_path"]).resolve()})', '', '- [ ] 내용 일치 / 전사 누락 없음 / 경계 일치', '']
        review += [f'{t["speaker"]}: {t["text"]}  ' for t in r['turns']]
        review.append('')
    (ROOT/'data/DIALOGUE_REVIEW.md').write_text('\n'.join(review))
    print(json.dumps(dict(candidates=len(rows), durations_seconds=[round((r['last_frame_exclusive']-r['first_frame'])/rate,3) for r in rows],
                         total_words=sum(len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",r['text'])) for r in rows),
                         verified=0, exp1_status='blocked_pending_content_and_boundary_verification'),indent=2))


if __name__ == '__main__':
    main()
