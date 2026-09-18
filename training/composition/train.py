#!/usr/bin/env python3
"""Train a small, source-validated composition classifier on CPU only.

Reuse the established intent model's feature encoding and optimizer math.
The original intent artifacts are never modified by this training program.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import struct
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('intent_training', HERE.parent/'intent/train.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SEED = 20260918
bag_features = base.feature_indices


def positioned_features(text):
    # Keep local language features, and distinguish the operation before the
    # first scalar from the operation after the second scalar. A bag of grams
    # alone cannot distinguish A's difference times B's sum from the reverse.
    position = 0
    parts = []
    for word in text.split():
        parts.append(f'{position}:{word}')
        position += word.count('#')
    return bag_features(text + ' | ' + ' '.join(parts))


base.feature_indices = positioned_features
CLASSES = [{'function':'defer', 'arity':0, 'arguments':[], 'templates':[]}]


def add(function, arity, arguments, templates):
    CLASSES.append(dict(function=function, arity=arity, arguments=arguments, templates=templates))


for power, noun in [(2, 'squares'), (3, 'cubes')]:
    for op, phrases in [
        ('add', ['add the', 'sum of', 'sum of the', 'the sum of']),
        ('subtract', ['difference of', 'the difference of', 'difference of the', 'the difference of the']),
        ('multiply', ['multiply the', 'product of', 'product of the', 'the product of']),
        ('mean', ['mean of', 'average of', 'mean of the', 'the mean of']),
    ]:
        add(op, 2, [f'({{a}})^{power}', f'({{b}})^{power}'],
            [f'{phrase} {noun} of {{a}} and {{b}}' for phrase in phrases])
    add('sum', 3, [f'[({{a}})^{power},({{b}})^{power},({{c}})^{power}]'],
        [f'sum of {noun} of {{a}}, {{b}}, and {{c}}', f'sum of the {noun} of {{a}} and {{b}} and {{c}}',
         f'the sum of {noun} of {{a}}, {{b}} and {{c}}', f'the sum of the {noun} of {{a}}, {{b}}, and {{c}}'])

add('subtract', 3, ['{c}', '({a})*({b})'], ['subtract the product of {a} and {b} from {c}', 'subtract product of {a} and {b} from {c}', 'subtract the product of {a} and {b} from the number {c}'])
add('divide', 3, ['{a}', '({b})*({c})'], ['divide {a} by the product of {b} and {c}', 'divide {a} by product of {b} and {c}', 'divide the number {a} by the product of {b} and {c}'])
add('add', 3, ['({a})*({b})', '{c}'], ['add the product of {a} and {b} to {c}', 'add product of {a} and {b} to {c}', 'add the product of {a} and {b} to the number {c}'])
add('subtract', 3, ['{c}', '({a})+({b})'], ['subtract the sum of {a} and {b} from {c}', 'subtract sum of {a} and {b} from {c}', 'subtract the sum of {a} and {b} from the number {c}'])
add('multiply', 4, ['({a})-({b})', '({c})+({d})'], ['multiply the difference of {a} and {b} by the sum of {c} and {d}', 'multiply difference of {a} and {b} by sum of {c} and {d}', 'multiply the difference of {a} and {b} by sum of {c} and {d}'])
add('multiply', 4, ['({a})+({b})', '({c})-({d})'], ['multiply the sum of {a} and {b} by the difference of {c} and {d}', 'multiply sum of {a} and {b} by difference of {c} and {d}', 'multiply the sum of {a} and {b} by difference of {c} and {d}'])
add('divide', 4, ['({a})+({b})', '({c})-({d})'], ['divide the sum of {a} and {b} by the difference of {c} and {d}', 'divide sum of {a} and {b} by difference of {c} and {d}', 'divide the sum of {a} and {b} by difference of {c} and {d}'])
add('power', 2, ['({a})-({b})', '2'], ['square of the difference of {a} and {b}', 'the square of the difference of {a} and {b}', 'square of difference of {a} and {b}'])
add('subtract', 2, ['{b}', '{a}'], ['{a} less than {b}', 'take {a} away from {b}', '{a} fewer than {b}', 'take the number {a} away from {b}'])

WRAPPERS = [
    ['{}', 'what is {}', 'calculate {}', 'compute {}', 'find {}', 'please calculate {}', 'please find {}', 'could you calculate {}', 'can you compute {}', 'give me {}', 'tell me {}', 'show me {}'],
    ['please compute {}', 'could you compute {}', 'what\'s {}', 'how much is {}'],
    ['can you find {}', 'could you find {}', 'work out {}', 'I need {}'],
]
NEGATIVES = ['add {a} and {b}', 'subtract {a} from {b}', 'multiply {a} by {b}', 'divide {a} by {b}',
             'square root of the sum of {a} and {b}', 'sum of square roots of {a} and {b}',
             'cube root of the sum of {a} and {b}', 'add {a} squared and {b} cubed',
             'difference of {a} and {b}', 'product of {a} and {b}', 'mean of {a} and {b}',
             'square of the sum of {a} and {b}', 'add the cube root of {a} and the square root of {b}',
             'add the squares of {a} and {b} then divide by {c}', 'subtract the sum of {a} and {b} from the product of {c} and {d}',
             'multiply the difference of {a} and {b} by {c}', 'divide the difference of {a} and {b} by the sum of {c} and {d}',
             'sum of squares of {a} and {b} and {c} and {d}', 'product of cubes of {a} and {b} and {c}',
             'the sum of {a} and {b} is less than {c}', 'take {a} from the sum of {b} and {c}']
VOCABULARY = set(re.findall(r"[a-zA-Z']+", ' '.join(sum(WRAPPERS, []) + NEGATIVES + [t for c in CLASSES for t in c['templates']])))
VOCABULARY = {w.lower() for w in VOCABULARY} - {'a', 'b', 'c', 'd'}
# Articles occur independently of the placeholder letters.
VOCABULARY.add('a')


def extract(text):
    if not text or len(text) > 512 or not text.isascii():
        return None
    spans, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch.isspace() or ch == '?':
            i += 1
        elif ch == ',':
            if i and i+1 < len(text) and text[i-1].isdigit() and text[i+1].isdigit():
                return None
            i += 1
        elif ch.isdigit() or ch in '+-.':
            if i and not text[i-1].isspace() and text[i-1] != ',':
                return None
            match = base.NUMBER.match(text, i)
            if not match or (match.end() < len(text) and not (text[match.end()].isspace() or text[match.end()] in ',?')):
                return None
            spans.append((i, match.end()))
            i = match.end()
        elif ch.isalpha() or ch == "'":
            end = i+1
            while end < len(text) and (text[end].isalpha() or text[end] == "'"):
                end += 1
            if text[i:end].lower() not in VOCABULARY:
                return None
            i = end
        else:
            return None
    return spans if 2 <= len(spans) <= 4 else None


def corpus(split, seen):
    rng = random.Random(SEED+split)
    rows = []
    templates = [(label, template) for label, c in enumerate(CLASSES) if label for template in c['templates']]
    templates += [(0, template) for template in NEGATIVES]
    for label, template in templates:
        for wrapper in WRAPPERS[split]:
            for suffix in ['', '?', ' please', ' for me']:
                for layout in range(3):
                    values = {k: str(rng.randint(101+split*1000, 997+split*1000)) for k in 'abcd'}
                    for key in values:
                        values[key] = rng.choice(['', '', '-','+']) + values[key] + rng.choice(['', '.25', 'e-2'])
                    text = wrapper.format(template.format(**values))+suffix
                    if layout == 1: text = text.replace(' ', '  ')
                    if layout == 2: text = text.replace(' ', '\t')
                    key = base.masked(text).strip().lower()
                    if key in seen: continue
                    seen.add(key)
                    spans = extract(text)
                    effective = label if spans and len(spans) == CLASSES[label]['arity'] else 0
                    rows.append({'input': text, 'label': effective})
    rng.shuffle(rows)
    return rows


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int,default=40); args=parser.parse_args()
    started=time.monotonic(); seen=set()
    datasets={name:corpus(i,seen) for i,name in enumerate(['train','validation','test'])}
    base.OUTPUTS=len(CLASSES); base.CLASSES=[(c['function'],c['arity'],False) for c in CLASSES]; base.extract=extract
    for name,rows in datasets.items():
        (HERE/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    matrices={name:base.matrix(rows) for name,rows in datasets.items()}
    rng=np.random.default_rng(SEED)
    params=[rng.normal(0,.12,(2048,32)).astype(np.float32),np.full(32,.02,np.float32),rng.normal(0,.15,(32,len(CLASSES))).astype(np.float32),np.zeros(len(CLASSES),np.float32)]
    means=[np.zeros_like(p) for p in params]; variances=[np.zeros_like(p) for p in params]
    labels=np.asarray([r['label'] for r in datasets['train']]); vy=np.asarray([r['label'] for r in datasets['validation']])
    best=None; best_score=(-1,-1,float('-inf')); history=[]; step=0
    for epoch in range(args.epochs):
        order=rng.permutation(len(labels)); loss=0
        for offset in range(0,len(order),256):
            ix=order[offset:offset+256]; x=matrices['train'][ix]; y=labels[ix]
            probs,h=base.predict_matrix(x,params)
            loss-=float(np.log(np.maximum(probs[np.arange(len(ix)),y],1e-12)).sum())
            delta=probs.copy(); delta[np.arange(len(ix)),y]-=1; delta/=len(ix)
            delta*=np.where(y==0,1.4,1).astype(np.float32)[:,None]
            w1,b1,w2,b2=params; dh=(delta@w2.T)*(h>0)
            grads=[x.T@dh+1e-5*w1,dh.sum(axis=0),h.T@delta+1e-5*w2,delta.sum(axis=0)]
            step+=1; rate=.008 if epoch<20 else .003
            for p,g,m,v in zip(params,grads,means,variances):
                m*=.9; m+=.1*g; v*=.999; v+=.001*g*g
                p-=rate*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        vp=base.predict_matrix(matrices['validation'],params)[0]
        vloss=float(-np.log(np.maximum(vp[np.arange(len(vy)),vy],1e-12)).mean())
        metric=base.scores(datasets['validation'],vp,.98)
        score=(int(metric['fp']==0),metric['tp'],-vloss)
        if score>best_score: best_score=score; best=[p.copy() for p in params]; selected=epoch+1
        history.append({'epoch':epoch+1,'loss':loss/len(labels),'validation_loss':vloss,'correct':metric['tp'],'wrong':metric['fp']})
        print(json.dumps(history[-1]),flush=True)
    validation=base.predict_matrix(matrices['validation'],best)[0]
    thresholds=[base.scores(datasets['validation'],validation,t) for t in [.90,.95,.97,.98,.99]]
    threshold=next((r['threshold'] for r in thresholds if r['fp']==0 and r['tp']>=100),1.0)
    metrics={name:base.scores(rows,base.predict_matrix(matrices[name],best)[0],threshold) for name,rows in datasets.items()}
    w1,b1,w2,b2=best
    model=ROOT/'assets/composition.bin'
    model.write_bytes(b'OMNICMP1'+struct.pack('<8IQ',1,2048,32,len(CLASSES),1,4,2,0,SEED)+w1.T.astype('<f4').tobytes()+b1.astype('<f4').tobytes()+w2.T.astype('<f4').tobytes()+b2.astype('<f4').tobytes())
    base.save(HERE/'classes.json',[{k:v for k,v in c.items() if k!='templates'} for c in CLASSES])
    (HERE/'vocabulary.txt').write_text(' '.join(sorted(VOCABULARY))+'\n')
    parity=[]
    for text in ['add the squares of -6 and 8','sum of cubes of 2, 3, and 4','what is 19 less than 83','multiply the difference of 21 and 8 by the sum of 2 and 3','  subtract the product of +1.50 and -2e-3 from 77?\r\n','add the squares of 1,000 and 2','do not add the squares of 5 and 8']:
        spans=extract(text); row={'input':text,'spans':spans,'masked':base.masked(text)}
        if spans: row.update(probabilities=base.scalar_probs(text,best))
        parity.append(row)
    base.save(HERE/'parity.json',parity)
    base.save(HERE/'metadata.json',{'model_id':'omni-composition-v1','format':'OMNICMP1','threshold':threshold,'validation_thresholds':thresholds,'seed':SEED,'outputs':len(CLASSES),'model_bytes':model.stat().st_size,'selected_epoch':selected,'epochs':args.epochs,'elapsed_seconds':time.monotonic()-started,'split_policy':'Disjoint wrappers; exact numeric-masked strings deduplicated across splits. Operation templates shared. Test never selects weights or threshold. Threshold calibrated on validation after an initial 0.98 trial. Not a broad English accuracy estimate.','metrics':metrics,'history':history,'hashes':{'model':base.sha(model),'script':base.sha(Path(__file__)),'feature_dependency':base.sha(HERE.parent/'intent/train.py'),**{name:base.sha(HERE/name) for name in ['classes.json','vocabulary.txt','parity.json','train.jsonl','validation.jsonl','test.jsonl']}}})
    print(json.dumps({'metrics':metrics,'bytes':model.stat().st_size,'selected_epoch':selected}),flush=True)


if __name__=='__main__': main()
