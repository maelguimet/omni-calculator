#!/usr/bin/env python3
"""Verify synthetic call schemas, arity, exact split separation and token bounds."""
import argparse
import collections
import hashlib
import json
import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("OMP_NUM_THREADS","2")
ROOT=pathlib.Path(__file__).resolve().parent.parent


def source(query):
    if query.startswith("Calculation request: "):
        return query[len("Calculation request: "):].split("\nPrevious tool call was rejected:",1)[0]
    return query


def validate(value,schema,path="arguments"):
    kind=schema["type"]
    assert {"object":lambda:isinstance(value,dict),"array":lambda:isinstance(value,list),
            "string":lambda:isinstance(value,str),"integer":lambda:type(value)==int}[kind](),path
    if "enum" in schema: assert value in schema["enum"],(path,value)
    if kind=="object":
        props=schema.get("properties",{})
        assert set(schema.get("required",[]))<=value.keys(),path
        assert value.keys()<=props.keys(),(path,value.keys()-props.keys())
        for key,item in value.items():validate(item,props[key],path+"."+key)
    if kind=="array":
        for index,item in enumerate(value):validate(item,schema["items"],path+f"[{index}]")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=pathlib.Path)
    parser.add_argument("--tokens",action="store_true")
    parser.add_argument("--max-len",type=int,default=672)
    parser.add_argument("--heldout",default="heldout.jsonl")
    args=parser.parse_args()
    tools=json.loads((ROOT/"experiments/tools.json").read_text())
    schema={t["name"]:t["parameters"] for t in tools}
    arity={name:1 for name in schema["function"]["properties"]["function"]["enum"]}
    arity.update({name:2 for name in ["add","subtract","multiply","divide","power","mod","gcd","lcm","ncr","npr","log","percent","round","quantile","matmul","solve_linear","poisson_pmf","poisson_cdf"]})
    arity.update({name:3 for name in ["normal_pdf","normal_cdf","normal_quantile","binomial_pmf","binomial_cdf"]})
    tok=None
    if args.tokens:
        from needle.model import tokenizer,finetune
        tokenizer.TOKENIZER_DIR=str(ROOT/"training/base/tokenizer")
        tokenizer.TOKENIZER_PREFIX=str(ROOT/"training/base/tokenizer/tokenizer")
        tok=tokenizer.get_tokenizer(8192)
    seen={};report={"max_len":args.max_len,"splits":{}}
    for split in ["train","heldout"]:
        path=args.directory/(args.heldout if split=="heldout" else "train.jsonl")
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        sources=set();lengths=[]
        for index,row in enumerate(rows):
            assert row["tools"]==tools,(split,index,"schema differs")
            assert row["system"]==(ROOT/"experiments/system.txt").read_text().strip()
            assert isinstance(row["query"],str) and row["query"].strip()
            assert len(row["answers"])<=1,(split,index,"multiple calls")
            for call in row["answers"]:
                assert set(call)=={"name","arguments"},(split,index)
                validate(call["arguments"],schema[call["name"]])
                if call["name"]=="function":
                    assert len(call["arguments"]["arguments"])==arity[call["arguments"]["function"]],(split,index,"arity")
                if call["name"]=="algebra":
                    a=call["arguments"]
                    assert ("lower" in a)==("upper" in a),(split,index,"bounds")
                    if a["operation"]=="root":assert "lower" in a
                    if a["operation"]=="substitute":assert "value" in a
            if tok:
                prompt,target=finetune.render_example(row)
                length=len(tok.encode(prompt))+len(tok.encode(target))+2;lengths.append(length)
                assert length<=args.max_len,(split,index,length)
            sources.add(source(row["query"]))
        seen[split]=sources
        report["splits"][split]={"count":len(rows),"unique_source_queries":len(sources),
                                  "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "max_tokens":max(lengths) if lengths else None,
                                  "families":dict(collections.Counter(r["_family"] for r in rows))}
    overlap=seen["train"]&seen["heldout"]
    report["exact_source_overlap"]=len(overlap)
    report["overlap_examples"]=sorted(overlap)[:10]
    (args.directory/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    assert not overlap,sorted(overlap)[:5]
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
