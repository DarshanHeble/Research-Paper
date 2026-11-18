"""Local proxy score for AI-style prose and source overlap in a .tex paper.

NOT Turnitin/GPTZero. It counts surface patterns that AI-detectors and
editors flag, plus verbatim 7-word overlap with the cited sources in pdfs/.
Lower is better. Also guards that numbers and cite keys did not change.

Usage: python3 style_score.py ieee_2547119.tex [--baseline base.json] [--save base.json]
"""
import json, re, statistics, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
AI_WORDS = """additionally furthermore moreover notably crucial crucially delve leverage leverages robust
seamless seamlessly pivotal underscore underscores highlight highlights landscape comprehensive
intricate showcase showcases foster fostering enhance enhances valuable vital realm paramount
streamline utilize utilizes facilitate facilitates holistic nuanced multifaceted testament tapestry
navigate cutting-edge essential ensure ensures ensuring precisely directly explicitly genuine
genuinely critical critically significant significantly substantially effectively inherently
fundamentally ultimately therefore thus hence deliberately comfortably""".split()
AI_PHRASES = ["taken together", "rather than", "it is worth", "plays a", "in order to",
              "not only", "serves as", "stands as", "a growing body", "first-class",
              "in the direction", "as posed", "by construction", "in turn", "as such",
              "the single largest", "real, measured", "best-effort"]
ING_TAIL = re.compile(r",\s+(highlighting|underscoring|ensuring|reflecting|showcasing|emphasizing|"
                      r"demonstrating|indicating|confirming|suggesting|making|giving|leaving|allowing)\b")
TRIAD = re.compile(r"\b[\w-]+(?:\s[\w-]+){0,3},\s[\w-]+(?:\s[\w-]+){0,3},\s(?:and|or)\s")
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def body_text(tex: str) -> str:
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    tex = tex.split(r"\begin{abstract}", 1)[1] if r"\begin{abstract}" in tex else tex
    tex = tex.split(r"\bibliographystyle", 1)[0]
    for env in ["figure", "figure*", "table", "table*", "tikzpicture", "axis", "equation"]:
        tex = re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (re.escape(env), re.escape(env)), " ", tex, flags=re.S)
    tex = re.sub(r"\\(cite|ref|label)\{[^}]*\}", " ", tex)
    tex = re.sub(r"\$[^$]*\$", " X ", tex)
    tex = re.sub(r"\\(texttt|emph|textbf|textit)\{([^}]*)\}", r"\2", tex)
    tex = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", tex)
    tex = tex.replace("~", " ").replace("{", " ").replace("}", " ").replace("``", '"').replace("''", '"')
    return re.sub(r"\s+", " ", tex).strip()


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", t) if len(s.split()) > 3]


def source_shingles(n=7):
    cache = HERE / ".source_shingles.json"
    if cache.exists():
        return set(map(tuple, json.load(open(cache))))
    sh = set()
    for p in sorted((HERE / "pdfs").iterdir()):
        txt = p.read_text(errors="ignore") if p.suffix == ".txt" else subprocess.run(
            ["pdftotext", str(p), "-"], capture_output=True, text=True).stdout
        w = re.findall(r"[a-z0-9]+", txt.lower())
        sh.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    json.dump([list(s) for s in sh], open(cache, "w"))
    return sh


def score(path):
    tex = Path(path).read_text()
    t = body_text(tex)
    words = t.split()
    nw = len(words)
    low = t.lower()
    sents = sentences(t)
    lens = [len(s.split()) for s in sents]
    k = 1000 / nw
    m = {
        "words": nw,
        "ai_words_per_1k": sum(len(re.findall(r"\b%s\b" % re.escape(w), low)) for w in AI_WORDS) * k,
        "ai_phrases_per_1k": sum(low.count(p) for p in AI_PHRASES) * k,
        "ing_tails_per_1k": len(ING_TAIL.findall(low)) * k,
        "triads_per_1k": len(TRIAD.findall(low)) * k,
        "semicolons_per_1k": t.count(";") * k,
        "colon_in_prose_per_1k": len(re.findall(r"[a-z]: [a-z]", t)) * k,
        "dashes_per_1k": (t.count("---") + t.count("\u2014")) * k,
        "parens_per_1k": t.count("(") * k,
        "mean_sent_len": statistics.mean(lens),
        "sent_len_cv": statistics.pstdev(lens) / statistics.mean(lens),
        "long_sents_pct": 100 * sum(l > 40 for l in lens) / len(lens),
    }
    sh = source_shingles()
    w = re.findall(r"[a-z0-9]+", low)
    grams = [tuple(w[i:i + 7]) for i in range(len(w) - 6)]
    m["source_overlap_pct"] = 100 * sum(g in sh for g in grams) / max(1, len(grams))
    # Composite: lower is better. Burstiness (cv) rewarded up to 0.6.
    m["SCORE"] = round(
        2.0 * m["ai_words_per_1k"] + 2.0 * m["ai_phrases_per_1k"] + 3.0 * m["ing_tails_per_1k"]
        + 1.0 * m["triads_per_1k"] + 1.0 * m["semicolons_per_1k"] + 0.5 * m["colon_in_prose_per_1k"]
        + 3.0 * m["dashes_per_1k"] + 0.5 * m["parens_per_1k"] + 0.5 * m["long_sents_pct"]
        + 40 * max(0.0, 0.6 - m["sent_len_cv"]) + 10 * m["source_overlap_pct"], 2)
    guard = {"numbers": sorted(set(NUM.findall(re.sub(r"(?<!\\)%.*", "", tex.split(r"\bibliographystyle")[0])))),
             "cites": sorted(set(k for c in re.findall(r"\\cite\{([^}]*)\}", tex) for k in c.split(",")))}
    return m, guard


if __name__ == "__main__":
    m, guard = score(sys.argv[1])
    for k, v in m.items():
        print(f"{k:24s} {v:.2f}" if isinstance(v, float) else f"{k:24s} {v}")
    if "--save" in sys.argv:
        json.dump(guard, open(sys.argv[sys.argv.index("--save") + 1], "w"))
    if "--baseline" in sys.argv:
        base = json.load(open(sys.argv[sys.argv.index("--baseline") + 1]))
        lost_n = sorted(set(base["numbers"]) - set(guard["numbers"]), key=float)
        lost_c = sorted(set(base["cites"]) - set(guard["cites"]))
        print("GUARD lost numbers:", lost_n or "none")
        print("GUARD lost cites:  ", lost_c or "none")
