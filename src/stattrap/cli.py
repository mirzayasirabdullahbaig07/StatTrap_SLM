import argparse
from pathlib import Path


def _items_args(p, defaults=True):
    p.add_argument("--per-family", type=int, default=30, help="base items per family")
    p.add_argument("--perms", type=int, default=2, help="option orders per base item")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--framing", type=int, default=0, choices=[0, 1, 2], help="question wrapper (robustness check)")
    p.add_argument("--families", nargs="*", help="subset of families (default: all)")


def _build_items(a):
    from .tasks import make_items
    return make_items(a.per_family, a.seed, a.perms, a.framing, a.families or None)


def main(argv=None):
    p = argparse.ArgumentParser(prog="stattrap", description="StatTrap-SLM: statistical traps for small language models")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("audit-items", help="STEP 0: check the benchmark cannot be solved by surface cues"); _items_args(s)
    s.add_argument("--out", default="results/item_audit.csv")

    s = sub.add_parser("export-items", help="write the item set as JSONL + print its hash"); _items_args(s)
    s.add_argument("--out", default="results/items.jsonl")

    s = sub.add_parser("verify-sheet", help="CSV for a human statistician to answer blind"); _items_args(s)
    s.add_argument("--n-per-family", type=int, default=4); s.add_argument("--out", default="results/verification_sheet.csv")

    r = sub.add_parser("run", help="evaluate models (resumable)"); _items_args(r)
    r.add_argument("--backend", choices=["ollama", "hf", "mock"], default="ollama")
    r.add_argument("--models", nargs="+", required=True); r.add_argument("--styles", nargs="+", default=["direct", "cot"])
    r.add_argument("--temperatures", nargs="+", type=float, default=[0.0]); r.add_argument("--decode-seeds", nargs="+", type=int, default=[0])
    r.add_argument("--max-new-tokens", type=int, default=512); r.add_argument("--load-in-4bit", action="store_true")
    r.add_argument("--limit", type=int, help="first N items only (smoke test)"); r.add_argument("--out", default="results/run.csv")

    a = sub.add_parser("analyze", help="tables + figures from a results CSV (item set read from its manifest)")
    a.add_argument("--csv", required=True)

    e = sub.add_parser("error-sheet", help="sample wrong CoT answers for manual error coding")
    e.add_argument("--csv", required=True); e.add_argument("--n", type=int, default=60)

    c = sub.add_parser("compare", help="paired comparison of two runs on the same items (e.g. framing 0 vs 2)")
    c.add_argument("--a", required=True); c.add_argument("--b", required=True)
    args = p.parse_args(argv)

    import pandas as pd
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

    if args.cmd == "audit-items":
        from .item_audit import audit_items, trap_position_check
        items = _build_items(args); t = audit_items(items)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True); t.round(4).to_csv(args.out, index=False)
        print(t.pivot(index="family", columns="heuristic", values="acc").round(2).to_string())
        print("\nanswer/trap letter shares:\n", trap_position_check(items).round(3))
        bad = t[t.flagged]
        print(f"\nmax heuristic accuracy {t.acc.max():.3f} (chance 0.25). Flagged (>=0.40 & p<1e-3): "
              f"{'NONE' if bad.empty else bad[['family', 'heuristic', 'acc']].to_string(index=False)}")
        anti = t[t.anti_cue]
        if len(anti):
            print("WARN elimination cues (heuristic almost never correct; ceiling 33% if exploited):\n" +
                  anti[["family", "heuristic", "acc"]].to_string(index=False))
        raise SystemExit(1 if len(bad) else 0)

    if args.cmd == "export-items":
        from .tasks import export_items
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        items = _build_items(args); print(f"{len(items)} items -> {args.out}  hash={export_items(items, args.out)}")

    elif args.cmd == "verify-sheet":
        from .sheets import verification_sheet
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        t = verification_sheet(_build_items(args), args.n_per_family, args.seed); t.to_csv(args.out, index=False)
        print(f"{len(t)} items -> {args.out}. Give it to a statistician WITHOUT the KEY_ columns; later compute agreement with the key.")

    elif args.cmd == "run":
        from .backends import HFBackend, MockBackend, OllamaBackend
        from .manifest import build_manifest, write_manifest
        from .runner import run_eval
        from .tasks import items_hash
        items = _build_items(args)
        if args.limit:   # round-robin across families so a smoke test covers every family
            from itertools import chain, zip_longest
            by_fam = {}
            for it in items:
                by_fam.setdefault(it.family, []).append(it)
            items = [x for x in chain.from_iterable(zip_longest(*by_fam.values())) if x is not None][: args.limit]
        be = {"ollama": OllamaBackend, "hf": lambda: HFBackend(args.load_in_4bit), "mock": lambda: MockBackend("random")}[args.backend]()
        from .tasks import PROMPT_VERSION
        extra = {"prompt_version": PROMPT_VERSION, **({"ollama_version": be.version()} if args.backend == "ollama" else {})}
        ih = items_hash(items)
        write_manifest(args.out, build_manifest({**vars(args), "families": args.families, "limit": args.limit,
                                                 "temperatures": args.temperatures, "decode_seeds": args.decode_seeds},
                                                ih, len(items), extra))
        n = run_eval(items, args.models, args.styles, be, args.out, args.max_new_tokens, args.temperatures, args.decode_seeds)
        print(f"wrote {n} new rows to {args.out} (item-set hash {ih})")

    elif args.cmd in ("analyze", "error-sheet"):
        from . import analysis as an, report as rp
        from .manifest import read_manifest
        from .tasks import make_items, items_hash
        m = read_manifest(args.csv)
        if m is None:
            raise SystemExit(f"No manifest next to {args.csv}; it was not produced by `stattrap run`.")
        ip = m["item_params"]
        items = make_items(ip["per_family"], ip["seed"], ip["perms"], ip.get("framing", 0), ip.get("families") or None)
        if m.get("run_params") and m.get("n_items") == len(items) and items_hash(items) != m["items_hash"]:
            if not m["run_params"].get("limit") and "limit" not in str(m):
                raise SystemExit("Item set rebuilt from manifest does not match its hash: generator code changed since the run.")
        df = pd.read_csv(args.csv, keep_default_na=False)
        if df.empty:
            raise SystemExit(f"{args.csv} has no result rows (the run failed before writing any). Fix the run first.")
        out = Path(args.csv).with_suffix("")
        if args.cmd == "error-sheet":
            from .sheets import error_sheet
            t = error_sheet(df, items, args.n); t.to_csv(f"{out}_error_sheet.csv", index=False)
            from .sheets import ERROR_CATEGORIES
            print(f"{len(t)} wrong CoT answers -> {out}_error_sheet.csv. Fill `error_category` with one of {ERROR_CATEGORIES}.")
            return
        fam, cot = an.by_family(df), an.cot_vs_direct(df)
        tables = {"summary": an.summary(df), "by_family": fam, "position_bias": an.position_bias(df),
                  "consistency": an.consistency(df, items), "cot_vs_direct": cot, "cot_vs_direct_by_family": an.cot_vs_direct(df, by="family"),
                  "pairwise_direct": an.pairwise_models(df, "direct"), "scaling": an.scaling(df),
                  "robustness_temperature": an.robustness(df), "truncation": an.truncation_report(df)}
        for name, t in tables.items():
            print(f"\n== {name} ==\n{t.round(3).to_string(index=False) if len(t) else '(not enough data)'}")
            t.round(5).to_csv(f"{out}_{name}.csv", index=False)
        figdir = Path(f"{out}_figures"); figdir.mkdir(exist_ok=True)
        for style in sorted(df["style"].unique()):
            rp.fig_trap_heatmap(fam, figdir / f"trap_heatmap_{style}.png", style)
            rp.fig_position_bias(tables["position_bias"], figdir / f"position_bias_{style}.png", style)
        rp.fig_cot_gain(cot, figdir / "cot_gain.png"); rp.fig_scaling(tables["scaling"], figdir / "scaling.png")
        print(f"\nTables: {out}_*.csv   Figures: {figdir}/")

    elif args.cmd == "compare":
        from . import analysis as an
        t = an.compare_runs(pd.read_csv(args.a, keep_default_na=False), pd.read_csv(args.b, keep_default_na=False), "A", "B")
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
