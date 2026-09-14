#!/usr/bin/env python3
"""CPU-only training of a bounded verbal-operation neural tool caller."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import struct
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "4")
import numpy as np
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 20260918
FEATURES, HIDDEN = 2048, 32
CLASSES = [
    ("defer", 0, False), ("add", 2, False), ("subtract", 2, False),
    ("subtract", 2, True), ("multiply", 2, False), ("divide", 2, False),
    ("divide", 2, True), ("power", 2, False), ("sqrt", 1, False),
    ("cbrt", 1, False), ("factorial", 1, False), ("mod", 2, False),
    ("gcd", 2, False), ("lcm", 2, False), ("ncr", 2, False),
    ("ncr", 2, True), ("npr", 2, False), ("npr", 2, True),
    ("percent", 2, False), ("abs", 1, False),
]
OUTPUTS = len(CLASSES)
FNV_OFFSET, FNV_PRIME = 14695981039346656037, 1099511628211
MASK64, SALT = (1 << 64) - 1, 0x9E3779B97F4A7C15
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
VOCABULARY = set("""what is what's whats wat iz wht wut wot how much calculate compute evaluate calc please can could you find the result answer value of for and by from to with a an me tell give get number numbers add plus sum addition total subtract minus difference take away less subtraction multiply multiplied times product multiplication divide divided quotient division into power raised raise exponent square squre cube root sqrt cbrt factorial mod modulo remainder gcd greatest common divisor highest factor lcm least lowest multiple ncr choose combination combinations npr permute permutation permutations percent per cent abs absolute between also""".split())
CUES = set("""add plus sum addition total subtract minus difference away less subtraction multiply multiplied times product multiplication divide divided quotient division power raised raise exponent root sqrt cbrt factorial mod modulo remainder gcd divisor factor lcm multiple ncr choose combination combinations npr permute permutation permutations percent cent abs absolute""".split())

# Every split has distinct operation templates, in addition to distinct wrappers.
# Template identities are not sampled into multiple splits.
TEMPLATES = {
1: (["add {a} and {b}", "{a} plus {b}", "sum of {a} and {b}", "add {a} to {b}", "addition of {a} and {b}", "the sum of {a} with {b}"], ["the result of {a} plus {b}", "sum {a} and {b}"], ["addition between {a} and {b}", "add the numbers {a} and {b}"]),
2: (["{a} minus {b}", "subtract {a} and {b}", "difference of {a} and {b}", "the difference between {a} and {b}"], ["the result of {a} minus {b}"], ["difference between the numbers {a} and {b}"]),
3: (["subtract {a} from {b}", "subtract the number {a} from {b}", "subtract {a} from the number {b}"], ["subtract the value {a} from {b}"], ["subtract {a} from the value {b}"]),
4: (["multiply {a} and {b}", "{a} times {b}", "{a} multiplied by {b}", "product of {a} and {b}", "multiply {a} by {b}", "multiplication of {a} and {b}"], ["the product of {a} with {b}", "the result of {a} times {b}"], ["product between {a} and {b}", "multiply the numbers {a} and {b}"]),
5: (["divide {a} by {b}", "{a} divided by {b}", "quotient of {a} and {b}", "the quotient of {a} by {b}"], ["the result of {a} divided by {b}"], ["divide the number {a} by {b}"]),
6: (["divide {a} into {b}", "divide the number {a} into {b}", "divide {a} into the number {b}"], ["divide the value {a} into {b}"], ["divide {a} into the value {b}"]),
7: (["{a} to the power of {b}", "raise {a} to the power {b}", "{a} raised to {b}", "{a} raised to the power of {b}"], ["the number {a} to the power {b}"], ["raise the number {a} to the power of {b}"]),
8: (["square root of {a}", "sqrt of {a}", "the square root of {a}", "square root {a}"], ["square root of the number {a}", "squre root of {a}"], ["the square root of the value {a}", "sqrt {a}"]),
9: (["cube root of {a}", "cbrt of {a}", "the cube root of {a}", "cube root {a}"], ["cube root of the number {a}"], ["the cube root of the value {a}", "cbrt {a}"]),
10: (["factorial of {a}", "{a} factorial", "factorial {a}", "the factorial of {a}"], ["factorial of the number {a}"], ["factorial of the value {a}"]),
11: (["{a} modulo {b}", "{a} mod {b}", "remainder of {a} by {b}", "remainder of {a} modulo {b}"], ["the number {a} modulo {b}"], ["the value {a} mod {b}"]),
12: (["gcd of {a} and {b}", "greatest common divisor of {a} and {b}", "gcd {a} and {b}", "greatest common divisor between {a} and {b}"], ["the gcd of {a} with {b}"], ["greatest common divisor of the numbers {a} and {b}"]),
13: (["lcm of {a} and {b}", "least common multiple of {a} and {b}", "lcm {a} and {b}", "least common multiple between {a} and {b}"], ["the lcm of {a} with {b}"], ["least common multiple of the numbers {a} and {b}"]),
14: (["{a} choose {b}", "ncr of {a} and {b}", "ncr {a} and {b}", "combinations of {a} choose {b}"], ["the result of {a} choose {b}"], ["ncr of the numbers {a} and {b}"]),
15: (["choose {a} from {b}", "choose the number {a} from {b}", "choose {a} from the number {b}"], ["choose the value {a} from {b}"], ["choose {a} from the value {b}"]),
16: (["{a} permute {b}", "npr of {a} and {b}", "npr {a} and {b}", "permutations of {a} and {b}"], ["the result of {a} permute {b}"], ["npr of the numbers {a} and {b}"]),
17: (["permutations of {a} from {b}", "permutations of the number {a} from {b}", "permutations of {a} from the number {b}"], ["permutations of the value {a} from {b}"], ["permutations of {a} from the value {b}"]),
18: (["{a} percent of {b}", "{a}% of {b}", "{a} per cent of {b}", "the value of {a} percent of {b}"], ["{a} percent of the number {b}"], ["{a}% of the value {b}"]),
19: (["absolute value of {a}", "abs of {a}", "the absolute value of {a}", "abs {a}"], ["absolute value of the number {a}"], ["the absolute value of the value {a}"]),
}
WRAPPERS = [
    ["{}", "what is {}", "what's {}", "calculate {}", "compute {}", "please calculate {}", "can you calculate {}", "could you calculate {}", "please compute {}", "find {}", "give me {}", "tell me {}"],
    ["wat iz {}", "how much is {}", "please find {}", "what is the answer for {}"],
    ["wht is {}", "could you compute {}", "can you find {}", "give me the result of {}"],
]
NEGATIVES = [
    ["what is the weather for {a}", "I have {a} apples and {b} pears", "subtract {a} from", "divide {a} by", "{a} plus", "the square of {a}", "the cube of {a}", "{a} percent increase of {b}", "square root of {a} plus {b}", "add {a} and {b} then stop", "add {a} kg and {b} kg", "factorial of three", "mean of {a} and {b}", "do not add {a} and {b}", "sqrt of {a} meters", "multiply {a} and the missing number", "{a} power", "add {a}", "{a} divided into {b}", "remainder after {a} is divided by {b}"],
    ["show weather at {a}", "{a} dogs live near {b} cats", "subtract from {a}", "{a} minus the answer", "sqrt of the sum of {a} and {b}", "lcm of {a} and {b} seconds", "what percent change from {a} to {b}", "never multiply {a} and {b}", "{a} or {b}", "what is three plus {a}", "{a} minus minus {b}"],
    ["is flight {a} on time", "my age is {a} and my weight is {b}", "divide by {a}", "add the answer to {a}", "absolute value of {a} times {b}", "choose {a} from {b} people", "decrease {a} percent of {b}", "don't calculate {a} plus {b}", "{a} plus what", "what is {a} plus seven", "divide {a} divide {b}"],
]


def extract(text):
    if not text or len(text) > 512 or not text.isascii():
        return None
    spans, words, i, cue_count = [], [], 0, 0
    while i < len(text):
        ch = text[i]
        if ch.isspace() or ch == "?":
            i += 1
        elif ch == ',':
            if i and i + 1 < len(text) and text[i - 1].isdigit() and text[i + 1].isdigit():
                return None
            i += 1
        elif ch == '%':
            if not spans or spans[-1][1] != i:
                return None
            cue_count += 1
            i += 1
        elif ch.isdigit() or ch in '+-.':
            if i and not text[i - 1].isspace() and text[i - 1] != ',':
                return None
            match = NUMBER.match(text, i)
            if match is None:
                return None
            end = match.end()
            if end < len(text) and not (text[end].isspace() or text[end] in ',?%'):
                return None
            spans.append((i, end))
            if len(spans) > 2:
                return None
            i = end
        elif ch.isalpha() or ch == "'":
            end = i + 1
            while end < len(text) and (text[end].isalpha() or text[end] == "'"):
                end += 1
            word = text[i:end].lower()
            if word not in VOCABULARY:
                return None
            cue_count += int(word in CUES)
            words.append(word)
            i = end
        else:
            return None
    if not spans or cue_count != 1:
        return None
    return spans


def masked(text):
    # Used for all training rows, including lexical negatives. Runtime uses
    # precisely the same substitution after the stricter extractor accepts.
    return NUMBER.sub('#', text)


def feature_indices(text):
    raw = b'\xff' + text.strip(' \t\r\n').lower().encode('utf-8') + b'\xfe'
    found = set()
    for width in range(1, 5):
        initial = FNV_OFFSET ^ ((width * SALT) & MASK64)
        for start in range(len(raw) - width + 1):
            h = initial
            for byte in raw[start:start + width]:
                h = ((h ^ byte) * FNV_PRIME) & MASK64
            found.add(h & (FEATURES - 1))
    return sorted(found)


def feature_scale(n):
    return np.float32(1) / np.sqrt(np.float32(n))


def matrix(rows):
    data, indices, pointers = [], [], [0]
    for row in rows:
        idx = feature_indices(masked(row['input']))
        indices.extend(idx)
        data.extend([feature_scale(len(idx))] * len(idx))
        pointers.append(len(indices))
    return csr_matrix((np.asarray(data, np.float32), np.asarray(indices, np.int32), np.asarray(pointers, np.int32)), shape=(len(rows), FEATURES))


def corpus(split, used):
    rng = random.Random(SEED + split)
    rows = []
    def append(template, label, category):
        for wrapper in WRAPPERS[split]:
            for suffix in ['', '?', ' please']:
                for layout in range(3):
                    base = rng.choice([17, 23, 41, 53]) if split == 0 else rng.choice([1009, 1021, 1031, 1061]) if split == 1 else rng.choice([7001, 7013, 7019, 7027])
                    a, b = str(base), str(base + 2)
                    if rng.random() < .4:
                        a = rng.choice(['-', '+', '']) + a + rng.choice(['.25', 'e-3', '.0'])
                    if rng.random() < .4:
                        b = rng.choice(['-', '+', '']) + b + rng.choice(['.75', 'e2', '.0'])
                    text = wrapper.format(template.format(a=a, b=b)) + suffix
                    if layout == 1:
                        text = text.replace(' ', '  ')
                    elif layout == 2:
                        text = text.upper() if rng.random() < .5 else text.replace(' ', '\t')
                    key = masked(text).strip().lower()
                    if key in used:
                        continue
                    used.add(key)
                    spans = extract(text)
                    # A template whose scope exceeds the lexical gate is defer.
                    effective = label if spans and len(spans) == CLASSES[label][1] else 0
                    rows.append({'input': text, 'label': effective, 'intended_class': label, 'eligible': spans is not None, 'category': category, 'split': ['train','validation','test'][split]})
    for label, families in TEMPLATES.items():
        for index, template in enumerate(families[split]):
            append(template, label, f'class_{label}_template_{index}')
    for index, template in enumerate(NEGATIVES[split]):
        append(template, 0, f'defer_template_{index}')
    rng.shuffle(rows)
    return rows


def predict_matrix(x, params):
    w1, b1, w2, b2 = params
    hidden = np.maximum(x @ w1 + b1, 0)
    logits = hidden @ w2 + b2
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, hidden


def scores(rows, probs, threshold):
    prediction = probs.argmax(axis=1)
    confidence = probs[np.arange(len(rows)), prediction]
    expected = np.asarray([row['label'] for row in rows])
    eligible = np.asarray([bool(extract(row['input'])) and len(extract(row['input'])) == CLASSES[prediction[i]][1] for i, row in enumerate(rows)])
    selected = (prediction != 0) & (confidence >= threshold) & eligible
    correct = selected & (prediction == expected)
    wrong = selected & (prediction != expected)
    tp, fp = int(correct.sum()), int(wrong.sum())
    total_positive = int((expected != 0).sum())
    return {'count': len(rows), 'threshold': threshold, 'tp': tp, 'fp': fp, 'literal_positives': total_positive, 'precision': tp / (tp + fp) if tp + fp else 1.0, 'recall': tp / total_positive if total_positive else 0.0, 'classification_accuracy': float((prediction == expected).mean()), 'false_positives': [{'input': row['input'], 'expected': int(expected[i]), 'predicted': int(prediction[i]), 'confidence': float(confidence[i])} for i, row in enumerate(rows) if wrong[i]]}


def scalar_probs(text, params):
    indices = feature_indices(masked(text)); scale = float(feature_scale(len(indices)))
    w1, b1, w2, b2 = params
    hidden = []
    for j in range(HIDDEN):
        value = float(b1[j])
        for index in indices:
            value += float(w1[index, j]) * scale
        hidden.append(max(value, 0.0))
    logits = []
    for k in range(OUTPUTS):
        value = float(b2[k])
        for j in range(HIDDEN):
            value += hidden[j] * float(w2[j, k])
        logits.append(value)
    peak = max(logits)
    values = [math.exp(value - peak) for value in logits]
    total = sum(values)
    return [value / total for value in values]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--epochs', type=int, default=32); args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    used = set()
    datasets = {name: corpus(i, used) for i, name in enumerate(['train', 'validation', 'test'])}
    for name, rows in datasets.items():
        (HERE / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    matrices = {name: matrix(rows) for name, rows in datasets.items()}
    rng = np.random.default_rng(SEED)
    params = [rng.normal(0,.12,(FEATURES,HIDDEN)).astype(np.float32), np.full(HIDDEN,.02,np.float32), rng.normal(0,.15,(HIDDEN,OUTPUTS)).astype(np.float32), np.zeros(OUTPUTS,np.float32)]
    means = [np.zeros_like(p) for p in params]; variances = [np.zeros_like(p) for p in params]
    labels = np.asarray([row['label'] for row in datasets['train']], dtype=np.int64)
    initial = scores(datasets['validation'], predict_matrix(matrices['validation'], params)[0], .98)
    best, best_score, history, step, selected_epoch = None, (-1, -1, float('-inf')), [], 0, 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(labels)); loss = 0.0
        for offset in range(0, len(order), 256):
            ix = order[offset:offset+256]; x = matrices['train'][ix]; y = labels[ix]
            probs, h = predict_matrix(x, params)
            loss -= float(np.log(np.maximum(probs[np.arange(len(ix)),y],1e-12)).sum())
            delta = probs.copy(); delta[np.arange(len(ix)),y] -= 1; delta /= len(ix)
            delta *= np.where(y == 0, 1.4, 1.0).astype(np.float32)[:,None]
            w1,b1,w2,b2 = params
            dh = (delta @ w2.T) * (h > 0)
            gradients = [x.T @ dh + 1e-5*w1, dh.sum(axis=0), h.T @ delta + 1e-5*w2, delta.sum(axis=0)]
            step += 1; rate = .008 if epoch < 16 else .003
            for p,g,m,v in zip(params,gradients,means,variances):
                m *= .9; m += .1*g; v *= .999; v += .001*g*g
                p -= rate*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        validation = predict_matrix(matrices['validation'],params)[0]
        vy = np.asarray([row['label'] for row in datasets['validation']])
        vloss = float(-np.log(np.maximum(validation[np.arange(len(vy)),vy],1e-12)).mean())
        metric = scores(datasets['validation'], validation, .98)
        history.append({'epoch':epoch+1,'train_loss':loss/len(labels),'validation_loss':vloss,'precision_at_0.98':metric['precision'],'recall_at_0.98':metric['recall'],'false_positives_at_0.98':metric['fp']})
        # Correct high-confidence coverage is the actual routing objective.
        # Prefer checkpoints with no validation misroutes, then more accepted
        # correct calls; cross entropy breaks ties. No test score is consulted.
        selection_score = (int(metric['fp'] == 0), metric['tp'], -vloss)
        if selection_score > best_score:
            best_score, best, selected_epoch = selection_score, [p.copy() for p in params], epoch+1
        print(json.dumps(history[-1]),flush=True)
    validation = predict_matrix(matrices['validation'],best)[0]
    candidates = [scores(datasets['validation'],validation,t) for t in [.98,.99,.995,.999,.9995,.9999]]
    safe = [row for row in candidates if row['fp'] == 0 and row['tp'] >= 100]
    threshold = safe[0]['threshold'] if safe else 1.0
    metrics = {name:scores(rows,predict_matrix(matrices[name],best)[0],threshold) for name,rows in datasets.items()}
    w1,b1,w2,b2 = best
    model = ROOT / 'assets/intent.bin'
    model.write_bytes(b'OMNIINT1'+struct.pack('<8IQ',1,FEATURES,HIDDEN,OUTPUTS,1,4,1,0,SEED)+w1.T.astype('<f4').tobytes()+b1.astype('<f4').tobytes()+w2.T.astype('<f4').tobytes()+b2.astype('<f4').tobytes())
    (HERE/'vocabulary.txt').write_text(' '.join(sorted(VOCABULARY))+'\n')
    (HERE/'operation-cues.txt').write_text(' '.join(sorted(CUES))+'\n')
    parity_cases = ['what is the square root of 144','wat iz 17 plus 23','subtract -5 from +12.50','divide 3 into 12','what is 25% of 88','  ABS OF -1.20e-3?\r\n','add 1,000','sqrt of 3 m','add 3 and 5 then square','what is one plus two','what is 7+3','root of .25','💀 sqrt 4']
    parity = []
    for text in parity_cases:
        spans = extract(text)
        row = {'input':text,'spans':spans,'masked':masked(text)}
        if spans:
            idx=feature_indices(masked(text))
            row.update(indices=idx,scale_bits=int(np.asarray(feature_scale(len(idx))).view(np.uint32)),probabilities=scalar_probs(text,best))
        parity.append(row)
    save(HERE/'parity.json',parity)
    metadata={'format':'OMNIINT1','model_id':'omni-intent-v1','seed':SEED,'features':FEATURES,'hidden':HIDDEN,'outputs':OUTPUTS,'classes':[{'id':i,'function':name,'arity':arity,'reverse':reverse} for i,(name,arity,reverse) in enumerate(CLASSES)],'threshold':threshold,'training':{'optimizer':'Adam','epochs':args.epochs,'selected_epoch':selected_epoch,'checkpoint_selection':'Prefer zero validation routing errors at 0.98, then greatest correct coverage, then lowest cross entropy. Initial loss-selected checkpoint is retained in baseline-loss-selection and deferred all requests under the minimum-100-correct validation threshold rule; selection objective revised from validation behavior, not test outcomes.','batch_size':256,'cpu_only':True,'initial_validation':initial,'elapsed_seconds':time.monotonic()-start,'dataset_bytes':sum((HERE/f'{name}.jsonl').stat().st_size for name in datasets)},'split_policy':'Held-out operation templates, wrappers, number pools and typography. Deduplicated across all splits after numeric masking, trimming and lowercasing. Test never used for gradients or model/threshold selection, but initial baseline test metrics were visible before the validation-based checkpoint objective revision. Classifier does not see number values. Fresh audit is separate and created after final model selection.','masked_duplicates_between_splits':0,'metrics':metrics,'validation_thresholds':candidates,'history':history,'hashes':{'model_sha256':sha(model),'script_sha256':sha(Path(__file__)),**{f'{name}_sha256':sha(HERE/f'{name}.jsonl') for name in datasets}},'model_bytes':model.stat().st_size}
    save(HERE/'metadata.json',metadata)
    print(json.dumps({'threshold':threshold,'model_bytes':model.stat().st_size,'metrics':metrics}),flush=True)


if __name__ == '__main__':
    main()
