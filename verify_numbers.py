"""Independently recompute every macro from the raw result files and diff
against what figures.py wrote. Deliberately does not import figures.py."""
import json, glob, re, statistics, pathlib, sys

R = {}
for f in glob.glob("artifacts/results/*.json"):
    r = json.load(open(f))
    R[r["run_id"]] = r

def g(model, ds, head="A_baseline"):
    return R.get(f"{model}__{ds}__{head}")

MAIN = ["ag_news", "sst2", "yelp_polarity", "yelp_full"]
ABL  = ["sst2", "ag_news"]
VARIANTS = ["A_baseline","B_frozen_all","C_frozen_half","D_narrow","E_wide","F_deep","G_linear"]
mean = statistics.fmean

exp = {}
# sizes
exp["distilParams"] = f'{g("distilbert","ag_news")["params"]["total_params"]/1e6:.1f}'
exp["bertParams"]   = f'{g("bert","ag_news")["params"]["total_params"]/1e6:.1f}'
pd_ = g("distilbert","ag_news")["params"]["total_params"]; pb_ = g("bert","ag_news")["params"]["total_params"]
exp["paramReduction"] = f'{100*(1-pd_/pb_):.0f}'

# accuracy
accD = mean(g("distilbert",d)["metrics"]["accuracy"]*100 for d in MAIN)
accB = mean(g("bert",d)["metrics"]["accuracy"]*100 for d in MAIN)
exp["meanAccDistil"] = f"{accD:.2f}"; exp["meanAccBert"] = f"{accB:.2f}"
exp["meanAccGap"] = f"{accB-accD:.2f}"; exp["accRetention"] = f"{100*accD/accB:.1f}"
gaps = {d: (g("bert",d)["metrics"]["accuracy"]-g("distilbert",d)["metrics"]["accuracy"])*100 for d in MAIN}
exp["minAccGap"] = f"{min(gaps.values()):.2f}"; exp["maxAccGap"] = f"{max(gaps.values()):.2f}"
exp["bertWins"] = str(sum(1 for v in gaps.values() if v > 0.1))
exp["distilTiesOrWins"] = str(sum(1 for v in gaps.values() if v <= 0.1))

# efficiency
exp["throughputDistil"] = f'{mean(g("distilbert",d)["latency"]["throughput_samples_per_s"] for d in MAIN):.0f}'
exp["throughputBert"]   = f'{mean(g("bert",d)["latency"]["throughput_samples_per_s"] for d in MAIN):.0f}'
exp["throughputSpeedup"] = f'{mean(g("distilbert",d)["latency"]["throughput_samples_per_s"] for d in MAIN)/mean(g("bert",d)["latency"]["throughput_samples_per_s"] for d in MAIN):.2f}'
exp["gpuLatencyDistil"] = f'{mean(g("distilbert",d)["latency"]["latency_ms_p50"] for d in MAIN):.2f}'
exp["gpuLatencyBert"]   = f'{mean(g("bert",d)["latency"]["latency_ms_p50"] for d in MAIN):.2f}'
exp["gpuSpeedup"] = f'{mean(g("bert",d)["latency"]["latency_ms_p50"] for d in MAIN)/mean(g("distilbert",d)["latency"]["latency_ms_p50"] for d in MAIN):.2f}'
mD = mean(g("distilbert",d)["training"]["train_peak_gpu_mb"] for d in MAIN)
mB = mean(g("bert",d)["training"]["train_peak_gpu_mb"] for d in MAIN)
exp["memReduction"] = f"{100*(1-mD/mB):.0f}"
exp["trainMemDistil"] = f"{mD/1024:.2f}"; exp["trainMemBert"] = f"{mB/1024:.2f}"

# ablation
f1 = {v: mean(g("distilbert",d,v)["metrics"]["f1_macro"]*100 for d in ABL) for v in VARIANTS}
best = max(f1, key=f1.get)
exp["bestVariantFone"] = f"{f1[best]:.2f}"
exp["frozenGap"] = f'{f1["A_baseline"]-f1["B_frozen_all"]:.2f}'
exp["halfFrozenGap"] = f'{f1["A_baseline"]-f1["C_frozen_half"]:.2f}'
shapes = ["D_narrow","E_wide","F_deep","G_linear","A_baseline"]
exp["headSpread"] = f'{max(f1[k] for k in shapes)-min(f1[k] for k in shapes):.2f}'
exp["nRuns"] = str(len(R))

# epoch study
ep = [R[f"{m}__{d}__A_baseline__ep4"] for d in ABL for m in ("distilbert","bert")]
bests = [r["training"]["best_epoch_by_val_f1"] for r in ep]
exp["epochBestMin"] = str(min(bests)); exp["epochBestMax"] = str(max(bests))
exp["epochStudyRuns"] = str(len(ep))
exp["epochGainOverTwo"] = f'{max(max(x["val_f1_macro"] for x in r["training"]["per_epoch"])*100 - [x for x in r["training"]["per_epoch"] if x["epoch"]==2][0]["val_f1_macro"]*100 for r in ep):.2f}'

# web data
w = json.load(open("web/data/words.json"))["meta"]
exp["wordOverlap"] = f'{w["mean_overlap"]:.1f}'; exp["wordOverlapPct"] = f'{w["overlap_pct"]:.0f}'
exp["wordCka"] = f'{w["cka"]:.3f}'; exp["nWords"] = str(w["n_words"])
exp["ckaEmbeddings"] = f'{w["layer_profile"][0]["cka"]:.3f}'
exp["ckaTop"] = f'{w["layer_profile"][-1]["cka"]:.3f}'
exp["purityBert"] = f'{w["purity"]["bert"]*100:.0f}'; exp["purityDistil"] = f'{w["purity"]["distilbert"]*100:.0f}'
_sent = json.load(open("web/data/sentences.json"))["datasets"]
exp["sentCka"] = f'{_sent["ag_news"]["meta"]["cka"]:.3f}'
exp["sentCkaMax"] = f'{max(v["meta"]["cka"] for v in _sent.values()):.3f}'
exp["sentCkaMin"] = f'{min(v["meta"]["cka"] for v in _sent.values()):.3f}'

got = dict(re.findall(r"renewcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}",
                      pathlib.Path("artifacts/tables/numbers.tex").read_text()))

ok = bad = 0
for k, v in sorted(exp.items()):
    if k not in got:
        print(f"  MISSING in numbers.tex: {k}"); bad += 1
    elif got[k] != v:
        print(f"  MISMATCH {k}: numbers.tex={got[k]!r}  independently computed={v!r}"); bad += 1
    else:
        ok += 1
print(f"\n{ok} macros verified against the raw result files, {bad} problems")
print(f"best variant independently: {best}   numbers.tex says: {got.get('bestVariant')}")
sys.exit(1 if bad else 0)
