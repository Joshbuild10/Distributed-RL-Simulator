"""The Streamlit explorer runs end to end: default settings, and a model that misses most windows."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.mark.slow
@pytest.mark.parametrize("model", [None, "Llama4-402B-A17B"])
def test_explorer_section_2(model):
    at = AppTest.from_file("../analysis/app.py", default_timeout=600)
    at.run()
    if model:
        next(s for s in at.selectbox if s.label == "Model").set_value(model).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.value.startswith("#### 2a") for m in at.markdown)
