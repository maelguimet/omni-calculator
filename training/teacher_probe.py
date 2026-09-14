#!/usr/bin/env python3
"""Bounded CPU-only comparison of QAT teacher-forced tokens with native calls."""
import argparse
import json
import os
import pathlib
import pickle
import resource
import time

os.environ["JAX_PLATFORMS"]="cpu"
os.environ["OMP_NUM_THREADS"]="2"
os.environ["OPENBLAS_NUM_THREADS"]="2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
resource.setrlimit(resource.RLIMIT_CPU,(240,250))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("adapter",type=pathlib.Path)
    parser.add_argument("cases",type=pathlib.Path)
    parser.add_argument("output",type=pathlib.Path)
    parser.add_argument("--greedy",action="store_true")
    parser.add_argument("--gpu",action="store_true")
    parser.add_argument("--encoding",choices=("upstream","contiguous"),default="upstream")
    args=parser.parse_args();started=time.perf_counter()
    if args.gpu:
        os.environ["JAX_PLATFORMS"]="cuda"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.25"
    import numpy as np
    import jax
    import jax.numpy as jnp
    from needle.model import tokenizer
    from needle.model.run import load_checkpoint
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.finetune import render_example,merge_lora
    from needle.model.quantize import configure_deploy,cq_ste_mixed_params,parse_bits_map
    root=pathlib.Path(__file__).resolve().parent
    tokenizer.TOKENIZER_DIR=str(root/"base/tokenizer")
    tokenizer.TOKENIZER_PREFIX=str(root/"base/tokenizer/tokenizer")
    params,config=load_checkpoint(str(root/"base/checkpoints/needle2.pkl"))
    config.dtype="float32"
    params=jax.tree.map(lambda v:jnp.asarray(v,dtype=jnp.float32),params)
    adapter=pickle.loads(args.adapter.read_bytes())
    lora={tuple(key.split("/")):{"A":jnp.asarray(v["A"]),"B":jnp.asarray(v["B"])} for key,v in adapter["lora"].items()}
    params=merge_lora(params,lora,adapter["scale"])
    params=cq_ste_mixed_params(params,*parse_bits_map(adapter["qat_bits_map"]))
    configure_deploy(act_bits=config.act_bits,kv_bits=config.kv_bits)
    model=SimpleAttentionNetwork(config);tok=tokenizer.get_tokenizer(config.vocab_size)
    rows=[json.loads(x) for x in args.cases.read_text().splitlines()]
    data=[]
    for row in rows:
        prompt,target=render_example(row)
        # Match upstream _encode exactly: target-alone encoding inserts its
        # own SentencePiece dummy-space token, unlike encoding concatenation.
        encoded=(tok.encode(prompt)+tok.encode(target) if args.encoding=="upstream"
                 else tok.encode(prompt+target))
        ids=[tokenizer.BOS_ID]+encoded+[tokenizer.EOS_ID]
        start=len([tokenizer.BOS_ID]+tok.encode(prompt))
        data.append((row,ids,start))
    maxlen=max(len(ids) for _,ids,_ in data)
    @jax.jit
    def predict(ids):
        logits=model.apply({"params":params},ids,quant=True)
        return jnp.argmax(logits,axis=-1),jax.nn.log_softmax(logits,axis=-1)
    results=[]
    print(f"{jax.default_backend()} probe: {len(data)} rows, sequence length {maxlen}, compiling",flush=True)
    for row,ids,start in data:
        padded=np.zeros((1,maxlen),dtype=np.int32);padded[0,:len(ids)]=ids
        predicted,logprob=predict(jnp.asarray(padded));predicted=np.asarray(predicted)[0];logprob=np.asarray(logprob)[0]
        mistakes=[]
        for position in range(start,len(ids)):
            expected=ids[position];actual=int(predicted[position-1])
            if actual!=expected:
                mistakes.append({"target_position":position-start,"expected":tok.decode([expected]),
                                 "predicted":tok.decode([actual]),"expected_id":expected,"predicted_id":actual,
                                 "expected_probability":float(np.exp(logprob[position-1,expected])),
                                 "predicted_probability":float(np.exp(logprob[position-1,actual])),
                                 "prefix":tok.decode(ids[start:position])})
        item={"query":row["query"],"answers":row["answers"],"target_tokens":len(ids)-start,"mistakes":mistakes}
        item["minimum_target_probability"]=float(min(np.exp(logprob[p-1,ids[p]]) for p in range(start,len(ids))))
        results.append(item)
        print(json.dumps({"query":row["query"],"mistakes":mistakes[:2]}),flush=True)
    if args.greedy:
        capacity=max(start for _,_,start in data)+96
        padded=np.zeros((len(data),capacity),dtype=np.int32)
        lengths=np.asarray([start for _,_,start in data],dtype=np.int32)
        for i,(_,ids,start) in enumerate(data):padded[i,:start]=ids[:start]
        generated=[[] for _ in data];finished=[False for _ in data]
        @jax.jit
        def next_token(tokens,positions):
            logits=model.apply({"params":params},tokens,quant=True)
            return jnp.argmax(logits[jnp.arange(tokens.shape[0]),positions-1],axis=-1)
        print("Compiling batched causal greedy rollout",flush=True)
        for _ in range(96):
            chosen=np.asarray(next_token(jnp.asarray(padded),jnp.asarray(lengths)))
            for i,token in enumerate(chosen):
                if finished[i]:continue
                token=int(token);padded[i,lengths[i]]=token;lengths[i]+=1;generated[i].append(token)
                if token==tokenizer.EOS_ID:finished[i]=True
            if all(finished):break
        for result,ids in zip(results,generated):
            result["greedy_ids"]=ids;result["greedy_text"]=tok.decode(ids)
            print(json.dumps({"query":result["query"],"greedy":result["greedy_text"]}),flush=True)
    args.output.write_text(json.dumps({"adapter":str(args.adapter),"backend":jax.default_backend(),
                                      "encoding":args.encoding,
                                      "affinity":sorted(os.sched_getaffinity(0)),"seconds":time.perf_counter()-started,"results":results},indent=2)+"\n")


if __name__=="__main__":main()
