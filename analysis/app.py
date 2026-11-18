"""
app.py -- Interactive explorer: change the threat model in the sidebar, see every figure and table
recompute. Same figure functions and renderer as the paper and dashboard.

    python -m analysis explore          (or: streamlit run analysis/app.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from analysis import figures  # noqa: F401  (registers everything)
from analysis import render, sweep
from analysis.scenarios import BATCH_MODES, Options
from analysis.spec import REGISTRY, SECTION_NOTES, SECTIONS, Fig
from feasibility import FeasConfig, HIGH_END_GPUS, MODELS, config_lines, with_cfg

st.set_page_config(page_title="Distributed-RL explorer", layout="wide")
D = FeasConfig(target_c_rl=2.5e25)


@st.cache_resource(show_spinner=False, max_entries=512)
def build(name, key, _opts):
    """Figure/table `name` under options `key` (the hashable repr of _opts)."""
    item = REGISTRY[name][1](_opts)
    sweep.flush()                              # share solver results with later sessions and builds
    return item


@st.cache_data(show_spinner=False, max_entries=512)
def image(name, key, theme, _fig):
    return render.render(_fig, "png", theme, title=False, dpi=170)


with st.sidebar:
    st.markdown("### Distributed-RL explorer")
    pick = st.radio("Chart theme", ["auto", "light", "dark"], horizontal=True,
                    help="auto follows this page's theme, so the charts sit flush on it")
    theme = (st.context.theme.type or "light") if pick == "auto" else pick
    st.markdown("**Run options**")
    penalties = st.toggle("Rahman penalties", help="Straggler tails, goodput, sync quality (penalties.py)")
    batch_mode = st.selectbox("Batch per budget", list(BATCH_MODES), help="\n\n".join(
        f"**{k}**: {v}" for k, v in BATCH_MODES.items()))
    st.markdown("**Threat model**")
    over = {}
    model = st.selectbox("Model", list(MODELS), index=list(MODELS).index("Llama3-405B"))
    gpu = st.selectbox("GPU", list(HIGH_END_GPUS), index=list(HIGH_END_GPUS).index(D.gpu))
    node = st.select_slider("Inference GPUs per node", [8, 16, 32, 64, 72, 128], value=D.gpus_per_node)
    wan = st.select_slider("WAN link (Mbps)", [100, 250, 500, 1_000, 2_500, 10_000, 100_000, 1_000_000],
                           value=int(D.wan_mbps))
    comp = st.select_slider("Broadcast compression", [1, 2, 4, 8, 16, 32, 64, 150], value=int(D.compression))
    distributed = st.toggle("Distributed trainer", value=D.train_distributed,
                            help="Trainer nodes at separate sites, all-reducing gradients every step over the WAN")
    gcomp = st.select_slider("Gradient compression", [1, 4, 16, 50, 150, 500], value=int(D.grad_compression),
                             disabled=not distributed)
    er = st.select_slider("Mean response length E[R]", [2_000, 5_000, 10_000, 20_000, 40_000, 80_000],
                          value=int(D.er))
    omega = st.select_slider("Oversampling", [1.0, 1.25, 1.5, 2.0, 3.0, 4.0], value=D.omega)
    stock = st.select_slider("GPU stock", [10_000, 100_000, 500_000, 1_500_000, 5_000_000], value=D.stock_gpus)
    for k, v, d in (("model", model, "Llama3-405B"), ("gpu", gpu, D.gpu), ("gpus_per_node", node, D.gpus_per_node),
                    ("wan_mbps", float(wan), D.wan_mbps), ("compression", float(comp), D.compression),
                    ("train_distributed", distributed, D.train_distributed),
                    ("grad_compression", float(gcomp), D.grad_compression), ("er", float(er), D.er),
                    ("omega", omega, D.omega), ("stock_gpus", stock, D.stock_gpus)):
        if v != d:
            over[k] = v
    opts = Options(penalties, batch_mode, tuple(sorted(over.items())))
    st.caption("Chart-specific levers win over these: a decentralised-vs-centralised chart sets its own link "
               "and node size, and validation anchors keep their published configs.")

sections = list(SECTIONS)
sec = st.segmented_control("Section", sections, default="2", format_func=lambda s: f"{s} · {SECTIONS[s]}")
if sec in SECTION_NOTES:
    st.markdown(SECTION_NOTES[sec])
names = [n for n, (s, _) in REGISTRY.items() if s == sec]
key = repr(opts)
with st.expander("Configuration", expanded=False):
    cfg = with_cfg(D, **{k: (MODELS[v] if k == "model" else v) for k, v in over.items()})
    st.code("\n".join(config_lines(cfg)), language=None)

for name in names:
    with st.spinner(f"Computing {name}..."):
        item = build(name, key, opts)
    st.markdown(f"#### {item.id} &nbsp; {item.title}")
    if isinstance(item, Fig):
        st.image(image(name, key, theme, item), width="stretch")
        if item.caption:
            st.caption(item.caption)
        c1, c2, c3, _ = st.columns([1, 1, 1, 5])
        # rendered on click, not on every rerun: a PDF costs ~3 s a figure
        c1.download_button("PDF", lambda it=item: render.render(it, "pdf", theme, title=False), f"{item.id}.pdf",
                           key=f"pdf-{name}")
        c2.download_button("PNG", lambda it=item: render.render(it, "png", theme, title=False), f"{item.id}.png",
                           key=f"png-{name}")
        c3.download_button("CSV", item.data.to_csv(index=False), f"{item.id}.csv", key=f"csv-{name}")
        with st.expander("data"):
            st.dataframe(item.data.drop(columns=[c for c in ("fallback",) if c in item.data]), hide_index=True)
    else:
        if item.caption:
            st.caption(item.caption)
        st.dataframe(item.data, width="stretch")
        st.download_button("CSV", item.data.to_csv(), f"{item.id}.csv", key=f"csv-{name}")
    st.divider()
