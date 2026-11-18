"""
web.py -- The static dashboard: every figure (one SVG per chart, from the same renderer as the paper),
its caption, its data and its paper downloads, grouped by section, with a light/dark toggle.

The toggle needs no second render: every colour in a chart is a theme ROLE, so each light-theme hex in
the SVG is rewritten to var(--role) and the page swaps the role values (theme.css_vars).
"""
import base64
import html
import os
import re

from . import render
from .spec import SECTIONS, Fig
from .theme import FONT, FONT_DIR, THEMES, css_vars

_HEX_TO_VAR = {v.lower(): f"var(--{k})" for k, v in THEMES["light"].items()}

CSS = """
*{box-sizing:border-box}html{color-scheme:light dark}
body{margin:0;font:14px/1.55 'Commissioner',system-ui,sans-serif;color:var(--fg);background:var(--bg);
display:grid;grid-template-columns:250px 1fr;min-height:100vh}
nav{position:sticky;top:0;height:100vh;overflow:auto;padding:22px 16px;border-right:1px solid var(--grid)}
nav h1{font-size:16px;margin:0;font-weight:700;color:var(--accent)}nav .sub{color:var(--muted);font-size:12px;margin:2px 0 14px}
nav h2{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:16px 0 4px}
nav a{display:block;color:var(--fg);text-decoration:none;padding:3px 8px;border-radius:6px;font-size:13px}
nav a:hover{background:var(--grid)}
button{font:inherit;font-size:12px;color:var(--fg);background:transparent;border:1px solid var(--spine);
border-radius:999px;padding:4px 12px;cursor:pointer;margin-bottom:6px}
main{padding:30px 40px 60px;max-width:1180px}
main>h2{font-size:20px;margin:34px 0 4px;color:var(--fg)}main>h2 span{color:var(--accent)}
.cfg{font:12px/1.6 ui-monospace,Consolas,monospace;color:var(--muted);border:1px solid var(--grid);
border-radius:10px;padding:12px 16px;white-space:pre-wrap}
.card{border:1px solid var(--grid);border-radius:12px;padding:18px 20px;margin:16px 0}
.card svg{width:100%;height:auto;display:block}
.card h3{margin:0 0 6px;font-size:15px}.card h3 b{color:var(--accent)}
.cap{color:var(--muted);font-size:13px;margin:8px 0 6px}
.dl a{font-size:12px;margin-right:12px;color:var(--accent);text-decoration:none}
details{margin-top:6px}summary{cursor:pointer;color:var(--muted);font-size:12px}
.tbl{overflow-x:auto}table{border-collapse:collapse;font-size:12px;margin-top:8px;white-space:nowrap}
th{font-weight:600;color:var(--muted)}th,td{border-bottom:1px solid var(--grid);padding:4px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
"""

JS = """
const r=document.documentElement,b=document.getElementById('tt');
const set=t=>{if(t)r.dataset.theme=t;else delete r.dataset.theme;b.textContent=(t||'auto')+' theme';
try{t?localStorage.setItem('theme',t):localStorage.removeItem('theme')}catch(e){}};
let s=null;try{s=localStorage.getItem('theme')}catch(e){};set(s);
b.onclick=()=>set({null:'light',light:'dark',dark:null}[r.dataset.theme||null]);
"""


def _fonts():
    faces = []
    for fname, weight in (("Commissioner-Regular.ttf", 400), ("Commissioner-SemiBold.ttf", 600),
                          ("Commissioner-Bold.ttf", 700)):
        with open(os.path.join(FONT_DIR, fname), "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        faces.append(f"@font-face{{font-family:'{FONT}';font-weight:{weight};"
                     f"src:url(data:font/ttf;base64,{b64}) format('truetype')}}")
    return "".join(faces)


def themed_svg(fig):
    """The figure's light SVG with every role colour rewritten to its CSS variable."""
    svg = render.render(fig, "svg", "light", title=False).decode("utf-8")
    svg = svg[svg.index("<svg"):]
    svg = re.sub(r'\s(width|height)="[\d.]+pt"', "", svg, count=2)      # let CSS size it
    return re.sub(r"#[0-9a-fA-F]{6}\b", lambda m: _HEX_TO_VAR.get(m.group(0).lower(), m.group(0)), svg)


def card(item, files_dir="."):
    """One figure or table as an HTML card (built in a worker process, assembled by page())."""
    head = f'<h3><b>{item.id}</b> &nbsp;{html.escape(item.title)}</h3>'
    cap = f'<p class="cap">{html.escape(item.caption)}</p>' if item.caption else ""
    if isinstance(item, Fig):
        dl = " ".join(f'<a href="{files_dir}/figures/{item.id}.{e}">{e.upper()}</a>' for e in ("pdf", "svg", "png"))
        dl += f' <a href="{files_dir}/tables/{item.id}.csv">CSV</a>'
        body = (f"{themed_svg(item)}{cap}<div class='dl'>{dl}</div>"
                f"<details><summary>data</summary><div class='tbl'>{_data_html(item)}</div></details>")
    else:
        body = f"{cap}<div class='tbl'>{item.data.to_html(border=0, escape=True)}</div>"
    return f'<section class="card" id="{item.id}">{head}{body}</section>'


def _data_html(fig):
    keep = [c for c in fig.data.columns if c not in ("fallback",)]
    return fig.data[keep].to_html(border=0, index=False, float_format=lambda v: f"{v:.4g}", na_rep="")


def page(entries, config_lines):
    """entries: [(section, id, title, card_html)] in display order."""
    by_sec = {}
    for sec, *rest in entries:
        by_sec.setdefault(sec, []).append(rest)
    nav, main = [], []
    for sec, its in by_sec.items():
        nav.append(f"<h2>{sec} &middot; {SECTIONS[sec]}</h2>"
                   + "".join(f'<a href="#{i}">{i} &nbsp;{html.escape(t[:46])}{"..." if len(t) > 46 else ""}</a>'
                             for i, t, _ in its))
        main.append(f"<h2><span>{sec}</span> &nbsp;{SECTIONS[sec]}</h2>" + "".join(h for _, _, h in its))
    cfg = html.escape("\n".join(config_lines))
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>Distributed-RL results</title>"
            f"<style>{_fonts()}{css_vars()}{CSS}</style></head><body>"
            f"<nav><h1>Distributed RL</h1><p class='sub'>Feasibility results &middot; <code>python -m analysis</code></p>"
            f"<button id='tt'>auto theme</button>{''.join(nav)}</nav>"
            f"<main><h2>Configuration</h2><div class='cfg'>{cfg}</div>{''.join(main)}</main>"
            f"<script>{JS}</script></body></html>")
