"""
web.py -- The dashboard: every figure's inline SVG (same renderer as the paper), its caption, its data
table, and links to the paper exports. One static HTML file, no JavaScript.
"""
import html

from . import render
from .figures import table

CSS = """
:root{--fg:#1a1a1a;--mut:#666;--line:#e4e4e4;--bg:#fafafa;--card:#fff;--acc:#0072B2}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
color:var(--fg);background:var(--bg);display:grid;grid-template-columns:220px 1fr}
nav{position:sticky;top:0;height:100vh;overflow:auto;padding:20px 16px;border-right:1px solid var(--line);background:var(--card)}
nav h1{font-size:15px;margin:0 0 4px}nav p{color:var(--mut);font-size:12px;margin:0 0 16px}
nav a{display:block;color:var(--fg);text-decoration:none;padding:4px 8px;border-radius:6px;font-size:13px}
nav a:hover{background:#eef4fa;color:var(--acc)}
main{padding:28px 36px;max-width:1100px}
.cfg{font:12px/1.6 ui-monospace,Consolas,monospace;color:var(--mut);background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:12px 16px;white-space:pre-wrap;margin-bottom:24px}
.fig{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-bottom:22px}
.fig svg{width:100%;height:auto}.cap{color:var(--mut);font-size:13px;margin:6px 0 10px}
.dl a{font-size:12px;margin-right:12px;color:var(--acc)}
details{margin-top:8px}summary{cursor:pointer;color:var(--mut);font-size:12px}
table{border-collapse:collapse;font-size:12px;margin-top:8px}
th,td{border-bottom:1px solid var(--line);padding:4px 10px;text-align:right}th:first-child,td:first-child{text-align:left}
"""


def page(figs, config_lines, paper_dir="paper", table_dir="tables"):
    nav = "".join(f'<a href="#{f.id}">{f.id} &middot; {html.escape(f.title)}</a>' for f in figs)
    cards = []
    for f in figs:
        links = " ".join(f'<a href="{paper_dir}/{f.id}.{ext}">{ext.upper()}</a>' for ext in ("pdf", "svg", "png"))
        cards.append(
            f'<section class="fig" id="{f.id}">{render.svg(f)}'
            f'<p class="cap">{html.escape(f.caption)}</p>'
            f'<div class="dl">{links} <a href="{table_dir}/{f.id}.csv">CSV</a></div>'
            f'<details><summary>Data</summary>{table(f).to_html(border=0)}</details></section>')
    cfg = html.escape("\n".join(config_lines))
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Distributed-RL results</title>"
            f"<style>{CSS}</style></head><body><nav><h1>Distributed-RL results</h1>"
            f"<p>Regenerate: <code>python -m analysis</code></p>{nav}</nav>"
            f"<main><div class='cfg'>{cfg}</div>{''.join(cards)}</main></body></html>")
