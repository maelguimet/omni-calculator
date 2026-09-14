#!/usr/bin/env python3
"""Fresh one-shot audit of frozen intent weights, never training or selecting."""
import json
from pathlib import Path
import random
import struct

from train import HERE, ROOT, CLASSES, FEATURES, HIDDEN, OUTPUTS, extract, masked, matrix, np, predict_matrix, save, scores, sha

CASES = [
    ("please give me the square root of 169", 8),
    ("could you please compute the cube root of -343", 9),
    ("please give me 31 plus 47", 1),
    ("could you please calculate 43 minus 19", 2),
    ("please subtract 17 from 91", 3),
    ("what's the product of 11 and 29", 4),
    ("please give me 81 divided by 9", 5),
    ("can you divide 7 into 84", 6),
    ("what is 13 to the power of 3", 7),
    ("give me the factorial of 9", 10),
    ("please give me 71 modulo 12", 11),
    ("please find gcd of 84 and 126", 12),
    ("could you calculate lcm of 21 and 35", 13),
    ("please give me 11 choose 3", 14),
    ("can you choose 4 from 13", 15),
    ("please give me 10 permute 3", 16),
    ("can you calculate permutations of 3 from 10", 17),
    ("could you please calculate 12.5 percent of 320", 18),
    ("what's the absolute value of -0.125", 19),
    ("what is the weather in Paris at 17", 0),
    ("subtract 5 meters from 19 meters", 0),
    ("sqrt of 3kg", 0), ("what is the square root of -16 m", 0),
    ("sqrt of twelve", 0), ("add twelve to 17", 0),
    ("divide 8 by zero", 0), ("square root of 16 plus 9", 0),
    ("multiply 11 by 3 and add 7", 0), ("add 17 and 3 then stop", 0),
    ("sqrt of (17+8)", 0), ("do not add 5 and 11", 0),
    ("never subtract 5 from 13", 0), ("factorial of 8 then email me", 0),
    ("what is 17 divided by", 0), ("subtract from 31", 0),
    ("what is 13 plus what", 0), ("the factorial of", 0),
    ("gcd of 5 and", 0), ("what is the square of 7", 0),
    ("what is the cube of 4", 0), ("3 percent change from 81", 0),
    ("increase 20 percent of 90", 0), ("what is 1,000 plus 2", 0),
    ("multiply 1.2.3 and 7", 0), ("add 1e and 3", 0),
    ("add + 3 and 5", 0), ("add 3+5", 0), ("sqrt(121)", 0),
    ("add 3 and 5; remove files", 0), ("sqrt of 9 degrees", 0),
    ("what is 30 percent of 50 euros", 0), ("mean of 3 and 7", 0),
    ("add 9 and 3 minus", 0), ("the root of 81", 0),
]


def main():
    metadata=json.loads((HERE/'metadata.json').read_text())
    data=(ROOT/'assets/intent.bin').read_bytes()
    assert sha(ROOT/'assets/intent.bin')==metadata['hashes']['model_sha256']
    assert data[:8]==b'OMNIINT1' and struct.unpack('<8I',data[8:40])==(1,FEATURES,HIDDEN,OUTPUTS,1,4,1,0)
    values=np.frombuffer(data[48:],dtype='<f4');a=FEATURES*HIDDEN;b=a+HIDDEN;c=b+HIDDEN*OUTPUTS
    params=[values[:a].reshape(HIDDEN,FEATURES).T,values[a:b],values[b:c].reshape(OUTPUTS,HIDDEN).T,values[c:]]
    seen=set()
    for name in ['train','validation','test']:
        seen.update(masked(json.loads(line)['input']).strip().lower() for line in (HERE/f'{name}.jsonl').read_text().splitlines())
    rng=random.Random(20260919)
    rows=[]
    for text,label in CASES:
        variants=[text, text.upper(), text.replace(' ','  '), text.replace(' ','\t')]
        for variant in variants:
            key=masked(variant).strip().lower()
            overlap=key in seen
            rows.append({'input':variant,'label':label,'eligible':extract(variant) is not None,'masked_overlap_with_prior':overlap,'category':'fresh_curated','split':'audit'})
    rng.shuffle(rows)
    probs=predict_matrix(matrix(rows),params)[0]
    labels=probs.argmax(axis=1)
    for i,row in enumerate(rows):
        row.update(predicted_class=int(labels[i]),confidence=float(probs[i,labels[i]]))
    path=HERE/'audit.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    new_indices=[i for i,row in enumerate(rows) if not row['masked_overlap_with_prior']]
    report={'model_sha256':sha(ROOT/'assets/intent.bin'),'data_sha256':sha(path),'script_sha256':sha(Path(__file__)),'policy':'Fresh cases written after freezing model and threshold. No tuning against this audit. Typography variants are correlated, not independent samples. Some familiar templates overlap after masking; report those separately. Rust integration tests additionally apply the mandatory semantic validator.','overall':scores(rows,probs,metadata['threshold']),'new_masked_inputs':scores([rows[i] for i in new_indices],probs[new_indices],metadata['threshold']),'prior_masked_overlap':len(rows)-len(new_indices)}
    save(HERE/'audit-report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
