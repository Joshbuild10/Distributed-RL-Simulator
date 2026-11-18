"""
theme.py -- ERA house style: Commissioner, a light palette (paper) and a dark one (slides, posters).

Every colour a chart uses is a named ROLE, identical in both themes. That is what lets the dashboard
switch theme in CSS: web.py maps each light hex back to its role and swaps the values (see
css_vars). So roles must have distinct hex values within a theme -- checked at import.
"""
import os

from matplotlib import font_manager

FONT = "Commissioner"
FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
for _f in os.listdir(FONT_DIR):
    if _f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(FONT_DIR, _f))

THEMES = {
    # ERA light: grey -> slate -> red -> teal for ordered series (e.g. 30/90/180/720 days)
    "light": dict(
        bg="#ffffff", fg="#1e2329", muted="#5a6570", grid="#e6e8eb", spine="#9aa3ab",
        accent="#d61a2d",
        s0="#b7bec5", s1="#4d5d6c", s2="#d7192d", s3="#2d6e7c",      # ordinal series
        s4="#e8939a", s5="#1f2e3a", s6="#c9a66b",                    # extra categorical series
        hm0="#f2f4f6", hm1="#d9e0e5", hm2="#b9c5ce", hm3="#8fa1ae", hm4="#62788a", hm5="#3a4f61",
        infeas="#f2c4c8", nofit="#dcdfe2", over="#d7192e", under="#4d5d6d",
    ),
    # Dark: coral red on near-black, slate and cyan as the cool counterparts
    "dark": dict(
        bg="#17191e", fg="#e9ecef", muted="#9aa3ad", grid="#2a2e35", spine="#59616b",
        accent="#ee5350",
        s0="#5e6a75", s1="#8fa3b3", s2="#ef5350", s3="#4bb8cb",
        s4="#f48a80", s5="#9c3b3d", s6="#d9c27a",
        hm0="#22262c", hm1="#2f3640", hm2="#3f4b58", hm3="#566778", hm4="#728799", hm5="#93a8ba",
        infeas="#6b2f33", nofit="#33373d", over="#ef5351", under="#5e6a76",
    ),
}
ORDINAL = ["s0", "s1", "s2", "s3"]                         # ordered sweeps: light-to-strong, red third
CATEGORICAL = ["s1", "s2", "s3", "s0", "s4", "s5", "s6"]   # unordered series: slate, red, teal, ...
HEAT = ["hm0", "hm1", "hm2", "hm3", "hm4", "hm5"]          # discrete steps (readable, and CSS-mappable)

for _name, _t in THEMES.items():
    assert len(set(_t.values())) == len(_t), f"theme {_name}: two roles share a hex"
    assert set(_t) == set(THEMES["light"]), f"theme {_name}: roles differ from light"


def series_roles(n, ordinal=False):
    pool = ORDINAL if ordinal and n <= len(ORDINAL) else CATEGORICAL
    return [pool[i % len(pool)] for i in range(n)]


def rc(theme="light"):
    t = THEMES[theme]
    return {
        "font.family": FONT, "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.labelweight": "semibold", "axes.titleweight": "bold",
        "text.color": t["fg"], "axes.labelcolor": t["fg"], "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["fg"], "ytick.labelcolor": t["fg"],
        "axes.edgecolor": t["spine"], "axes.facecolor": t["bg"], "figure.facecolor": t["bg"],
        "savefig.facecolor": t["bg"], "legend.labelcolor": t["fg"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": t["grid"], "grid.linewidth": 0.6,
        "axes.axisbelow": True, "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT,
        "mathtext.bf": f"{FONT}:bold", "svg.fonttype": "none", "pdf.fonttype": 42,
    }


def css_vars():
    """:root custom properties for both themes, keyed by role (web.py swaps them on toggle)."""
    def block(t):
        return ";".join(f"--{k}:{v}" for k, v in THEMES[t].items())
    return (f":root{{{block('light')}}}"
            f"@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{{block('dark')}}}}}"
            f":root[data-theme=dark]{{{block('dark')}}}")
