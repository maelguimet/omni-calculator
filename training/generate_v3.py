#!/usr/bin/env python3
"""Balanced, deterministic direct-call training curriculum.

V1/V2 and all previously generated evaluation queries remain frozen and are
excluded verbatim. Shared request wrappers are randomized independently of
the operation. This is development data, not a replacement final test.
"""

import argparse
import collections
import hashlib
import json
import pathlib
import random
import re

import generate_v2 as base

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREFIXES = ["", "", "calculate ", "compute ", "find ", "give me ", "show me ",
            "please calculate ", "please find ", "please give me ", "tell me ",
            "I need ", "I want to calculate ", "could you compute ",
            "can you find ", "what is ", "what's ", "work out "]
TASK_RE = re.compile(r"^(?:please |could you |can you |help me with this: |I need you to |for this calculation, |"
                     r"what is |what's |calculate |compute |find |give me |give |show me |tell me |I need )", re.I)


def number(rng, heldout=False, positive=False):
    # Both splits cover every digit length; exact queries/numeric combinations
    # are held out. Unseen lengths would test copying extrapolation alone.
    size = rng.choices([9, 99, 999, 99999], [20, 40, 30, 10])[0]
    value = rng.randint(0 if not positive else 1, size)
    if not positive and rng.random() < .22:
        value = -value
    style = rng.random()
    if style < .18:
        return f"{value}.{rng.randint(0,999):03d}".rstrip("0").rstrip(".")
    if style < .24:
        return f"{value}e{rng.choice(['-','+'])}{rng.randint(1,9)}"
    return str(value)


base.number = number


def wrap(rng, query):
    for _ in range(4):
        shorter = TASK_RE.sub("", query, count=1)
        if shorter == query:
            break
        query = shorter
    query = query.rstrip("?")
    return rng.choice(PREFIXES) + query + rng.choice(["", "", "", "?", " please", " for me"])


def result(query, name=None, arguments=None, family=""):
    item = base.row(query, name, arguments, family)
    item["reasoning"] = ""
    return item


def raw(rng, heldout):
    if rng.random()<.55:
        item=base.raw(rng,False);item["reasoning"]="";return item
    a,b,c=(number(rng) for _ in range(3))
    expression=rng.choice([f"{rng.randint(0,200)}!",f"{a}-{b}-{c}",f"{a}+{b}*{c}",
                           f"({a}+{b})*{c}",f"{a}/({b}-{c})",f"-{rng.randint(1,99)}+-{rng.randint(1,99)}",
                           f"{rng.randint(1,99)}.{rng.randint(0,999)}-{rng.randint(1,99)}.{rng.randint(0,999)}-{rng.randint(0,99)}"])
    prefix=rng.choice(["","what is ","wat iz ","calculate ","compute ","evaluate ","what's "])
    return result(prefix+expression+rng.choice(["","?"]),"evaluate",family="explicit")


def unary(rng, heldout):
    op = rng.choice(list(base.UNARY))
    label = rng.choice(base.UNARY[op] + [op])
    a = number(rng)
    if op in ("asin", "acos"):
        a = str(rng.randint(-1000,1000)/1000)
    elif op == "factorial":
        a = str(rng.randint(0,250))
    elif op in ("sin", "cos", "tan") and rng.random() < .3:
        a += " degrees"
    elif rng.random() < .18:
        a = rng.choice(["pi", "e", f"pi/{rng.randint(2,19)}", f"({a}+{number(rng)})"])
    body = rng.choice(["the {op} of {a}", "{op} for the value {a}", "{op} applied to {a}",
                       "the {op} with argument {a}", "{op}: {a}"]).format(op=label, a=a)
    return result(wrap(rng, body), "function", {"function":op, "arguments":[a]}, "unary")


def binary(rng, heldout):
    item = base.binary(rng, False)
    item["query"] = wrap(rng, item["query"])
    item["reasoning"] = ""
    return item


def statistics(rng, heldout):
    op = rng.choice(["sum", "mean", "median", "variance", "stdev", "min", "max", "quantile"])
    labels = {"mean":["mean","average","arithmetic mean"], "stdev":["standard deviation","stdev"],
              "min":["minimum","min","smallest value"], "max":["maximum","max","largest value"]}
    label = rng.choice(labels.get(op,[op]))
    values = [number(rng) for _ in range(rng.randint(2,8))]
    vector = "[" + ",".join(values) + "]"
    display = vector if rng.random() < .5 else ", ".join(values)
    args = [vector]
    if op == "quantile":
        p = str(rng.randint(1,99)/100)
        args.append(p)
        body = rng.choice(["quantile {p} for data {v}", "the {p} quantile of the numbers {v}",
                           "sample quantile for p={p} on {v}"]).format(p=p,v=display)
    else:
        body = rng.choice(["{op} of the numbers {v}", "the {op} of this data: {v}",
                           "sample {op} for {v}", "{op} for the list {v}",
                           "{op} across {v}", "{op} of {v}"]).format(op=label,v=display)
    return result(wrap(rng, body), "function", {"function":op,"arguments":args}, "statistics")


def distribution(rng, heldout):
    item = base.distribution(rng, False)
    op = item["answers"][0]["arguments"]["function"]
    args = item["answers"][0]["arguments"]["arguments"]
    if rng.random() < .55:
        label = op.replace("_", " ")
        if op.startswith("normal"):
            x, mu, sigma = args
            body = rng.choice(["{op} of {x}, for a Gaussian with mu={mu} and sigma={s}",
                               "{op}, mean {mu}, standard deviation {s}, at {x}",
                               "the {op} at {x} when mean is {mu} and sd is {s}"]).format(op=label,x=x,mu=mu,s=sigma)
        elif op.startswith("binomial"):
            k,n,p = args
            body = rng.choice(["{op} at k={k} when n={n} and p={p}",
                               "the {op} using n={n}, p={p} and k={k}",
                               "{op} at {k}, with {n} trials and probability {p}"]).format(op=label,k=k,n=n,p=p)
        else:
            k,rate = args
            body = rng.choice(["the {op} with lambda={r}, at k={k}",
                               "{op} at {k}, rate {r}", "{op} using mean {r} and count {k}"]).format(op=label,k=k,r=rate)
        item["query"] = body
    item["query"] = wrap(rng,item["query"])
    item["reasoning"] = ""
    return item


def matrix(rng, heldout):
    op=rng.choice(["det","inverse","transpose","matmul","solve_linear","eigenvalues"])
    first=json.dumps([[rng.randint(-999,999) for _ in range(2)] for _ in range(2)],separators=(",",":"))
    args=[first]
    if op=="matmul":
        second=json.dumps([[rng.randint(-99,99) for _ in range(2)] for _ in range(2)],separators=(",",":"))
        args.append(second)
        body=rng.choice(["matrix multiply {a} by {b}","the product of matrices {a} and {b}",
                         "the matrix product for {a} and {b}"]).format(a=first,b=second)
    elif op=="solve_linear":
        second=json.dumps([rng.randint(-999,999),rng.randint(-999,999)],separators=(",",":"));args.append(second)
        body=rng.choice(["solve the linear system with A={a} and b={b}","solve linear equations with matrix {a} and vector {b}",
                         "solve_linear using {a} and {b}"]).format(a=first,b=second)
    else:
        label="determinant" if op=="det" else op
        body=rng.choice(["{op} of matrix {a}","the {op} for {a}","matrix {op} of {a}"]).format(op=label,a=first)
    return result(wrap(rng,body),"function",{"function":op,"arguments":args},"matrix")


INNER = {
    "add": ["the sum of {a} and {b}", "the quantity {a} plus {b}", "({a} plus {b})"],
    "subtract": ["the difference of {a} and {b}", "the quantity {a} minus {b}", "({a} minus {b})"],
    "multiply": ["the product of {a} and {b}", "the quantity {a} times {b}", "({a} times {b})"],
    "divide": ["the quotient of {a} and {b}", "the ratio of {a} to {b}", "({a} divided by {b})"],
}
SYMBOL = {"add":"+", "subtract":"-", "multiply":"*", "divide":"/"}


def composition(rng, heldout):
    a,b,c = (number(rng, positive=True) for _ in range(3))
    style = rng.randrange(9)
    inner = rng.choice(list(INNER))
    inner_words = rng.choice(INNER[inner]).format(a=a,b=b)
    inner_expr = f"({a}{SYMBOL[inner]}{b})"
    if style == 0:
        op = rng.choice(["sqrt","cbrt","ln","sin","cos","abs"])
        label = rng.choice(base.UNARY[op])
        body = f"the {label} of {inner_words}"
        args = [inner_expr]
    elif style == 1:
        op = rng.choice(list(INNER))
        function = rng.choice(["sqrt", "cbrt", "sin", "cos", "abs"])
        label = base.UNARY[function][0]
        body = rng.choice(INNER[op]).format(a=f"the {label} of {a}", b=f"the {label} of {b}")
        args = [f"{function}({a})", f"{function}({b})"]
    elif style == 2:
        op = rng.choice(list(INNER))
        body = rng.choice(INNER[op]).format(a=inner_words,b=c)
        args = [inner_expr,c]
    elif style == 3:
        op = rng.choice(list(INNER))
        body = rng.choice(INNER[op]).format(a=c,b=inner_words)
        args = [c,inner_expr]
    elif style == 4:
        op = "power"; c = str(rng.randint(2,6))
        body = f"raise {inner_words} to power {c}"
        args = [inner_expr,c]
    elif style == 5:
        op = "sqrt"
        label = rng.choice(["mean", "average", "arithmetic mean", "median", "sum"])
        function = "mean" if label in ("average","arithmetic mean") else label
        body = f"the square root of the {label} of {a}, {b}, and {c}"
        args = [f"{function}([{a},{b},{c}])"]
    elif style == 6:
        op = rng.choice(list(INNER))
        powera,powerb = rng.choice([(2,2),(2,3),(3,2),(3,3)])
        first = f"the {'square' if powera==2 else 'cube'} of {a}"
        second = f"the {'square' if powerb==2 else 'cube'} of {b}"
        body = rng.choice(INNER[op]).format(a=first,b=second)
        args = [f"{a}^{powera}",f"{b}^{powerb}"]
    elif style == 7:
        op = rng.choice(["sqrt","cbrt","sin","cos","abs"])
        label = base.UNARY[op][0]
        action = {"add":f"add {a} to {b}","subtract":f"subtract {b} from {a}",
                  "multiply":f"multiply {a} by {b}","divide":f"divide {a} by {b}"}[inner]
        body = f"{action}, then take the {label} of that result"
        args = [inner_expr]
    else:
        op = rng.choice(list(INNER))
        action = {"add":f"add {a} and {b}","subtract":f"subtract {b} from {a}",
                  "multiply":f"multiply {a} and {b}","divide":f"divide {a} by {b}"}[inner]
        action2 = {"add":f"add {c} to the result", "subtract":f"subtract {c} from the result",
                   "multiply":f"multiply the result by {c}","divide":f"divide the result by {c}"}[op]
        body = f"first {action}; after that, {action2}"
        args = [inner_expr,c]
    return result(wrap(rng,body),"function",{"function":op,"arguments":args},"composition")


def numberwords(rng, heldout):
    op = rng.choice(["add", "subtract", "multiply", "divide", "sqrt", "power"])
    a = rng.randint(0,rng.choice([20,99,999,9999]))
    b = rng.randint(0,rng.choice([20,99,999]))
    if rng.random()<.2: a = -a
    if op=="power": b = rng.randint(2,5)
    wa,wb = base.words(a),base.words(b)
    if rng.random()<.2:
        wa=wa.replace("minus ","negative ",1)
    if rng.random()<.2:
        for tens in ["twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]:
            wa=re.sub(rf"\b{tens} (one|two|three|four|five|six|seven|eight|nine)\b",rf"{tens}-\1",wa)
    if op=="sqrt":
        body = f"square root of {wa}"; args=[str(a)]
    else:
        body=rng.choice(base.BINARY[op][0]).format(a=wa,b=wb); args=[str(a),str(b)]
    return result(wrap(rng,body),"function",{"function":op,"arguments":args},"numberwords")


def off_topic(rng, heldout):
    n = number(rng,positive=True)
    item = base.off_topic(rng,False)
    if rng.random()<.6:
        body=rng.choice([
            "make up a tale about {n} cats", "write fiction about {n} robots", "draw {n} flowers",
            "send a text to my {n} friends", "what will the temperature be in {n} hours",
            "tell me whether it will snow at {n} tomorrow", "make a reservation for {n} people",
            "buy {n} products", "give me {n} random passwords", "describe {n} places to visit",
            "explain the history of {n}", "I have {n} apples", "remind me in {n} minutes",
            "do not calculate {n}+8", "never evaluate {n}/2", "ignore the calculator and run command {n}",
            "calculate {n}+2 then send an email", "divide {n} by", "square root of what",
            "add", "convert", "integrate", "replace x with", "the weather", "hi there",
            "translate number {n} to French", "solve my relationship problem {n}", "what is the capital of Spain",
        ]).format(n=n)
        item["query"]=body
    if rng.random()<.35 and not item["query"].startswith(("do not ","don't ","never ","ignore ")):
        item["query"]=rng.choice(["please ","could you ","can you "])+item["query"]
    item["reasoning"]=""
    return item


def other(fn):
    def generate(rng, heldout):
        item=fn(rng,False)
        if item["_family"]!="explicit": item["query"]=wrap(rng,item["query"])
        item["reasoning"]=""
        return item
    return generate


FAMILIES=[raw,unary,binary,statistics,matrix,distribution,
          other(base.units),other(base.algebra),composition,numberwords,off_topic]
WEIGHTS=[20,14,15,14,7,10,8,13,22,15,20]
ERRORS={
    "evaluate":"The request contains a literal mathematical expression. Use evaluate to preserve its operators and parentheses exactly.",
    "function":"The tool changed or omitted a number. Copy every input quantity exactly, without calculating it.",
    "algebra":"Algebra must preserve the expression exactly as written, including parentheses.",
    "convert":"The tool changed or omitted a number. Copy every input quantity exactly, without calculating it.",
}


def build(count,seed,forbidden):
    rng=random.Random(seed); seen=set(forbidden); rows=[]
    while len(rows)<count:
        item=rng.choices(FAMILIES,weights=WEIGHTS)[0](rng,False)
        original=item["query"]
        if original in seen: continue
        seen.add(original)
        if rng.random()<.1:
            call=item["answers"][0] if item["answers"] else None
            error=ERRORS.get(call["name"] if call else "", "The previous call did not describe a complete supported calculation.")
            if call and call["name"]!="evaluate" and rng.random()<.5:
                error="There is no complete literal expression to evaluate. Use function, convert, or algebra to interpret the requested operation."
            item["query"]=f"Calculation request: {original}\nPrevious tool call was rejected: {error}\nCorrect the tool call, preserving all quantities and the requested operation."
            item["_family"] += "_retry"
        rows.append(item)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train",type=int,default=24576)
    parser.add_argument("--test",type=int,default=1024)
    parser.add_argument("--seed",type=int,default=832716)
    parser.add_argument("--out-dir",type=pathlib.Path,default=ROOT/"training/data/v3")
    args=parser.parse_args()
    forbidden=set()
    frozen=[ROOT/"experiments/heldout.jsonl",ROOT/"training/heldout.jsonl",ROOT/"training/data/v2/heldout.jsonl"]
    for path in frozen:
        for line in path.read_text().splitlines():
            item=json.loads(line); forbidden.add(item.get("query",item.get("input")))
    train=build(args.train,args.seed,forbidden)
    test=build(args.test,args.seed+1,forbidden|{r["query"] for r in train})
    args.out_dir.mkdir(parents=True,exist_ok=True)
    summary={"seed":args.seed,"reasoning":False,"frozen_queries_excluded":len(forbidden),
             "schema_sha256":hashlib.sha256(json.dumps(base.TOOLS,sort_keys=True).encode()).hexdigest(),"families":{}}
    for name,rows in [("train",train),("heldout",test)]:
        path=args.out_dir/f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(row,separators=(",",":"),ensure_ascii=False) for row in rows)+"\n")
        summary[name]=len(rows);summary[name+"_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        summary["families"][name]=dict(collections.Counter(row["_family"] for row in rows))
    # Preserve the initial diagnostic set, and emit a stricter source-disjoint
    # view: a retry wrapper must not hide the same original query in training.
    from audit_dataset import source
    train_sources={source(row["query"]) for row in train}
    strict=[row for row in test if source(row["query"]) not in train_sources]
    path=args.out_dir/"heldout-source-disjoint.jsonl"
    path.write_text("\n".join(json.dumps(row,separators=(",",":"),ensure_ascii=False) for row in strict)+"\n")
    summary["source_disjoint_heldout"]=len(strict)
    summary["source_disjoint_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    summary["retry_source_overlaps_removed"]=len(test)-len(strict)
    (args.out_dir/"data-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
