"""H134 P5 drop-logging receipt for one JobSet: per-step completeness of xw/drops_<js>.jsonl (every step 1..STEPS present
once; duplicates are merged by step in xw2.py) and the global aggregation of dropped and routed assignments over the 15 MoE
layers. Any gap means "budget not shown" for that run.   python3 receipt_p5.py <js> <steps>"""
import json, sys
js, STEPS = sys.argv[1], int(sys.argv[2]); W = "/weka/oe-eval-default/abhishekr/watch/tpu/h134/xw"
r = [json.loads(l) for l in open(f"{W}/drops_{js}.jsonl")]
st = {x["step"]: x for x in r}
gaps = sorted(set(range(1, STEPS + 1)) - set(st)); extra = sorted(set(st) - set(range(1, STEPS + 1)))
dropped = sum(x["dropped"] for x in st.values()); routed = sum(x["routed"] for x in st.values())
worst = max((x["dropped"] / x["routed"] for x in st.values() if x["routed"]), default=0.0)
out = {"jobset": js, "steps_expected": STEPS, "steps_present": len(st), "records": len(r), "gaps": len(gaps), "first_gaps": gaps[:20],
       "out_of_range": extra[:20], "dropped": dropped, "routed": routed, "total_frac": dropped / routed if routed else None,
       "worst_step_frac": worst, "max_factor": max((x["max_factor"] for x in st.values()), default=None),
       "budget": "budget not shown" if gaps else ("within #550" if routed and dropped / routed <= 1e-4 and worst <= 0.01 else "OVER #550")}
json.dump(out, open(f"{W}/receipt_p5_{js}.json", "w"), indent=1); print(json.dumps({k: out[k] for k in ("jobset", "steps_present", "gaps", "total_frac", "worst_step_frac", "budget")}))
