"""Load raw TSVs, normalise name/address text, cache as parquet in work/."""
import os, re, sys, time
import polars as pl
from unidecode import unidecode

D = os.environ.get("DATA_DIR", "dataset/")
W = os.environ.get("WORK_DIR", "work/")
os.makedirs(W, exist_ok=True)

ABBR = {
    # address
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle", "pl": "place", "pkwy": "parkway",
    "hwy": "highway", "ter": "terrace", "trl": "trail", "sq": "square", "mkt": "market",
    "fl": "floor", "flr": "floor", "apt": "apartment", "ste": "suite", "bldg": "building",
    "no": "number", "nr": "near", "opp": "opposite", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "cdp": "", "twp": "township", "hno": "house", "h": "house",
    "nagr": "nagar", "clny": "colony", "sect": "sector", "sec": "sector", "chk": "chowk",
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai", "bangalore": "bengaluru",
    "gurgaon": "gurugram", "allahabad": "prayagraj", "ahmadabad": "ahmedabad",
    # names / legal
    "pvt": "private", "prvt": "private", "ltd": "limited", "lmtd": "limited", "corp": "corporation",
    "co": "company", "inc": "incorporated", "intl": "international", "mfg": "manufacturing",
    "svcs": "services", "svc": "services", "assoc": "associates", "bros": "brothers", "dept": "department",
    "univ": "university", "hosp": "hospital", "ctr": "center", "centre": "center", "tech": "technologies",
    "and": "", "the": "", "of": "",
}
US_STATES = dict(al="alabama", ak="alaska", az="arizona", ar="arkansas", ca="california", co_="colorado",
    ct_="connecticut", de="delaware", fl_="florida", ga="georgia", hi="hawaii", id="idaho", il="illinois",
    in_="indiana", ia="iowa", ks="kansas", ky="kentucky", la="louisiana", me="maine", md="maryland",
    ma="massachusetts", mi="michigan", mn="minnesota", ms="mississippi", mo="missouri", mt="montana",
    nv="nevada", nh="new hampshire", nj="new jersey", nm="new mexico", ny="new york", nc="north carolina",
    nd="north dakota", oh="ohio", ok="oklahoma", or_="oregon", pa="pennsylvania", ri="rhode island",
    sc="south carolina", sd="south dakota", tn="tennessee", tx="texas", ut="utah", vt="vermont",
    va="virginia", wa="washington", wv="west virginia", wi="wisconsin", wy="wyoming", dc="district columbia")
IN_STATES = dict(mh="maharashtra", up="uttar pradesh", wb="west bengal", tn_="tamil nadu", ka="karnataka",
    kl="kerala", ap="andhra pradesh", ts="telangana", tg="telangana", gj="gujarat", rj="rajasthan",
    mp="madhya pradesh", hr="haryana", pb="punjab", br="bihar", dl="delhi", od="odisha", or_="odisha",
    orissa="odisha", jh="jharkhand", cg="chhattisgarh", uk="uttarakhand", hp="himachal pradesh",
    ga_="goa", as_="assam", jk="jammu kashmir")

# France overrides (st/ste mean saint/sainte there; French street words and stopwords)
FR = {
    "st": "saint", "ste": "sainte", "r": "rue", "bd": "boulevard", "bld": "boulevard", "bvd": "boulevard",
    "av": "avenue", "ave": "avenue", "rte": "route", "ch": "chemin", "chem": "chemin", "pl": "place",
    "imp": "impasse", "all": "allee", "sq": "square", "qu": "quai", "fbg": "faubourg", "fg": "faubourg",
    "crs": "cours", "pass": "passage", "res": "residence", "resid": "residence", "lot": "lotissement",
    "ham": "hameau", "bat": "batiment", "bt": "batiment", "esc": "escalier", "apt": "appartement",
    "zi": "zone industrielle", "za": "zone artisanale", "zac": "zone", "ctre": "centre", "center": "centre",
    "cedex": "", "bp": "", "cs": "", "cie": "compagnie", "ets": "etablissements", "etab": "etablissements",
    "grd": "grand", "gd": "grand", "gde": "grande", "pte": "petite", "pt": "petit", "mal": "marechal",
    "gal": "general", "gen": "general", "pdt": "president", "prof": "professeur", "dr": "docteur",
    "de": "", "du": "", "des": "", "la": "", "le": "", "les": "", "l": "", "d": "", "et": "",
    "au": "", "aux": "", "en": "", "sur": "",
}

FR.update({"ndeg": "", "n": "", "no": "", "numero": ""})  # "N° 19" -> "Ndeg 19" after unidecode
# French legal forms, also dotted/spaced (S.A.R.L., S A S), written as one token (sarl, sas, ...). Kept in the name like
# US "inc"/"llc": a different legal form is evidence against a match (decoys reuse the name with another legal form).
def _dotted(w):
    return r"\s?".join(c + r"\.?" for c in w)
FR_LEGAL = [(r"\b(?:" + _dotted(w) + r")(?:\b|$)", " " + w + " ")
            for w in ("sarl", "sasu", "sas", "eurl", "sci", "snc", "sa", "ei")]  # longest first: sasu before sas
# French regions and departments: one source writes the region, another the department; drop a comma field that is one
FR_AREAS = set("""auvergne rhone alpes|bourgogne franche comte|bretagne|centre val de loire|corse|grand est|hauts de france
|ile de france|normandie|nouvelle aquitaine|occitanie|pays de la loire|provence alpes cote d azur|alsace|aquitaine|auvergne
|bourgogne|centre|champagne ardenne|franche comte|languedoc roussillon|limousin|lorraine|midi pyrenees|nord pas de calais
|basse normandie|haute normandie|picardie|poitou charentes|rhone alpes|paca|france|ain|aisne|allier|alpes de haute provence
|hautes alpes|alpes maritimes|ardeche|ardennes|ariege|aube|aude|aveyron|bouches du rhone|calvados|cantal|charente
|charente maritime|cher|correze|corse du sud|haute corse|cote d or|cotes d armor|creuse|dordogne|doubs|drome|eure|eure et loir
|finistere|gard|haute garonne|gers|gironde|herault|ille et vilaine|indre|indre et loire|isere|jura|landes|loir et cher|loire
|haute loire|loire atlantique|loiret|lot|lot et garonne|lozere|maine et loire|manche|marne|haute marne|mayenne
|meurthe et moselle|meuse|morbihan|moselle|nievre|nord|oise|orne|pas de calais|puy de dome|pyrenees atlantiques
|hautes pyrenees|pyrenees orientales|bas rhin|haut rhin|rhone|haute saone|saone et loire|sarthe|savoie|haute savoie
|seine maritime|seine et marne|yvelines|deux sevres|somme|tarn|tarn et garonne|var|vaucluse|vendee|vienne|haute vienne|vosges
|yonne|territoire de belfort|essonne|hauts de seine|seine saint denis|val de marne|val d oise""".replace("\n", "").split("|"))

def fr_name(e: pl.Expr) -> pl.Expr:
    e = e.str.to_lowercase().str.replace_all(r"['’]", " ")
    for pat, w in FR_LEGAL:
        e = e.str.replace_all(pat, w)
    return e

def fr_addr(e: pl.Expr) -> pl.Expr:
    norm = pl.element().str.to_lowercase().str.replace_all(r"['’]", " ").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()
    return (e.str.split(",").list.eval(pl.element().filter(~norm.is_in(list(FR_AREAS)))).list.join(",")
             .str.to_lowercase().str.replace_all(r"['’]", " ").str.replace_all(r"\b0+(\d)", "$1"))

# name-only tokens: honorifics, "trading as" / "doing business as" glue, web remnants
NAME_ONLY = {"sri": "", "shri": "", "shree": "", "smt": "", "www": "", "com": "", "dba": "", "trading": ""}

def base_map(country):
    return {**ABBR, **FR} if country == "France" else dict(ABBR)

def addr_map(country):
    m = base_map(country)
    st = US_STATES if country == "US" else IN_STATES if country == "India" else {}
    for k, v in st.items():
        m[k.rstrip("_")] = v
    m.update({k: v for k, v in EXTRA.get(country, {}).items() if " " not in k})
    return m

def phrase_map(e: pl.Expr, country) -> pl.Expr:
    """Multi-word abbreviations (e.g. 'h no') replaced before token mapping."""
    for k, v in EXTRA.get(country, {}).items():
        if " " in k:
            e = e.str.replace_all(r"\b" + k + r"\b", v)
    return e

F = os.environ.get("FRIENDS_DIR", "friends/returned/")

def read_tsv(name):
    p = F + name
    if not os.path.exists(p):
        return None
    return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)

def load_dicts():
    """Hand-made dictionaries from helpers (optional). Returns native word map, extra token maps per country."""
    native, extra = {}, {"India": {}, "France": {}}
    if os.path.exists(W + "native_map_learned.tsv"):  # learned from train matches by src/learn_native.py
        d = pl.read_csv(W + "native_map_learned.tsv", separator="\t", quote_char=None)
        native = dict(zip(d["a"], d["b"]))
    d = read_tsv("india_native_words.tsv")
    if d is not None:
        native.update({a: b.strip().lower() for a, b in zip(d["native_word"], d["english"]) if b.strip()})
    d = read_tsv("india_abbreviations.tsv")
    if d is not None:
        extra["India"] = {a.strip().lower(): b.strip().lower() for a, b in zip(d["short"], d["full"]) if a.strip() and b.strip()}
    for fn in ("france_name_tokens.tsv", "france_address_tokens.tsv"):
        d = read_tsv(fn)
        if d is not None:
            for a, b, ty in zip(d["token"], d["canonical"], d["type"]):
                if ty.strip() == "stop":
                    extra["France"][a] = ""
                elif b.strip():
                    extra["France"][a] = b.strip().lower()
    return native, extra

NATIVE, EXTRA = load_dicts()
NATIVE_WORD = re.compile(r"[ऀ-෿‌‍]+")
_multi = sorted((k for k in NATIVE if " " in k), key=len, reverse=True)
NATIVE_PHRASE = re.compile("|".join(map(re.escape, _multi))) if _multi else None

def to_ascii(x):
    if NATIVE:
        if NATIVE_PHRASE is not None:
            x = NATIVE_PHRASE.sub(lambda m: " " + NATIVE[m.group(0)] + " ", x)
        x = NATIVE_WORD.sub(lambda m: NATIVE.get(m.group(0), m.group(0)), x)
    return unidecode(x)

def translit(s: pl.Series) -> pl.Series:
    """unidecode only the non-ascii strings (much faster than all)."""
    s = s.str.replace_all("Â", " ")  # mojibake of non-breaking space
    mask = s.str.contains(r"[^\x00-\x7F]")
    idx = mask.arg_true()
    if len(idx) == 0:
        return s
    vals = [to_ascii(x) for x in s.gather(idx).to_list()]
    return s.scatter(idx, pl.Series(vals, dtype=pl.String))

def base_clean(e: pl.Expr) -> pl.Expr:
    return (e.str.to_lowercase()
             .str.replace_all(r"<null>|\bn/a\b|\bnull\b", " ")
             .str.replace_all(r"\bm\s*/\s*s\b\.?|\bwww\.|\.(com|net|org|co\.in|in)\b", " ")  # M/s prefix, web remnants
             .str.replace_all(r"&", " and ")
             .str.replace_all(r"'", "")
             .str.replace_all(r"(\d)(st|nd|rd|th)\b", "$1")
             .str.replace_all(r"[^a-z0-9]+", " ")
             .str.strip_chars())

def token_map(df: pl.DataFrame, col: str, mapping: dict) -> pl.Series:
    ks = list(mapping.keys()); vs = list(mapping.values())
    t = (df.select(pl.int_range(pl.len()).alias("_i"), pl.col(col).str.split(" ").alias("_t"))
           .explode("_t")
           .with_columns(pl.col("_t").replace(ks, vs))
           .filter(pl.col("_t").is_not_null() & (pl.col("_t") != ""))
           .group_by("_i", maintain_order=True).agg(pl.col("_t").str.join(" ")))
    out = pl.DataFrame({"_i": pl.int_range(df.height, eager=True)}).join(t, on="_i", how="left")
    return out["_t"].fill_null("")

def prep(split, src):
    t = time.time()
    p = f"{D}{split}/{split}_source{src}.tsv"
    df = pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
    df = df.with_columns(translit(df["business_name"]).alias("name_t"), translit(df["business_address"]).alias("addr_t"))
    df = df.with_columns(base_clean(pl.col("name_t")).alias("name_c"), base_clean(pl.col("addr_t")).alias("addr_c"))
    parts = []
    for c, g in df.group_by("country"):
        m = addr_map(c[0])
        if c[0] == "France":
            g = g.with_columns(base_clean(fr_name(pl.col("name_t"))).alias("name_c"), base_clean(fr_addr(pl.col("addr_t"))).alias("addr_c"))
        g = g.with_columns(phrase_map(pl.col("name_c"), c[0]), phrase_map(pl.col("addr_c"), c[0]))
        g = g.with_columns(token_map(g, "name_c", {**base_map(c[0]), **NAME_ONLY, **{k: v for k, v in EXTRA.get(c[0], {}).items() if " " not in k}}).alias("name_n"), token_map(g, "addr_c", m).alias("addr_n"))
        parts.append(g)
    df = pl.concat(parts).drop("name_t", "addr_t")
    df.write_parquet(f"{W}{split}_s{src}.parquet")
    print(split, src, df.shape, f"{time.time()-t:.0f}s", flush=True)

if __name__ == "__main__":
    splits = sys.argv[1:] or ["train", "test"]
    for sp in splits:
        for s in (1, 2, 3):
            prep(sp, s)
    gt = pl.read_csv(D + "train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False,
                     missing_utf8_is_empty_string=True)
    gt.write_parquet(W + "train_gt.parquet")
