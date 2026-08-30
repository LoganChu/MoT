"""Correctness gate for the Qwen3-VL ablations.

These assert on ``generate()`` rather than on a forward pass, and that is the
whole point. An earlier version of this file checked logits under ``model(**b)``
while the study ran ``model.generate(...)``; the patch it was validating was
never reached during generation, so every check passed against an intervention
that did nothing. A gate that does not exercise the production path proves
nothing, which is why ``test_intervention_changes_output`` exists at all.

Marked slow: each test needs a GPU and the 4.26 GB checkpoint.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pyarrow = pytest.importorskip("pyarrow.parquet")

from mot.ablate import AblationSpec, ablated
from mot.ablate_configs import ABLATIONS

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]


@pytest.fixture(scope="module")
def rig():
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import pyarrow.parquet as pq

    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16).cuda().eval()
    table = pq.ParquetFile(hf_hub_download(
        "lmms-lab/textvqa", "data/validation-00000-of-00003.parquet",
        repo_type="dataset")).read_row_group(0)
    rows = [(Image.open(io.BytesIO(table.column("image")[i].as_py()["bytes"])).convert("RGB"),
             table.column("question")[i].as_py()) for i in range(6)]

    def generate(i, spec=None, with_image=True):
        img, question = rows[i]
        content = ([{"type": "image"}] if with_image else []) + \
                  [{"type": "text", "text": question}]
        text = proc.apply_chat_template([{"role": "user", "content": content}],
                                        add_generation_prompt=True)
        kw = {"images": [img]} if with_image else {}
        batch = proc(text=[text], return_tensors="pt", **kw).to("cuda")
        model.model.rope_deltas = None
        with torch.no_grad():
            if spec is None:
                out = model.generate(**batch, max_new_tokens=24, do_sample=False)
            else:
                with ablated(model, spec):
                    out = model.generate(**batch, max_new_tokens=24, do_sample=False)
        return proc.decode(out[0][batch["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()

    return model, generate


MROPE_ARMS = [a for a in ABLATIONS.values() if a.mrope and not a.deepstack_off]
ALL_ARMS = [a for a in ABLATIONS.values() if a.name != "baseline"]


@pytest.mark.parametrize("spec", MROPE_ARMS, ids=lambda s: s.name)
def test_mrope_is_inert_without_an_image(rig, spec):
    """With no image every token already has t == h == w, so M-RoPE reduces to
    ordinary 1-D RoPE and no reinterpretation of those rows can change anything."""
    _, generate = rig
    assert not spec.touches_text_only_path()
    for i in range(3):
        assert generate(i, None, with_image=False) == generate(i, spec, with_image=False)


@pytest.mark.parametrize("spec", ALL_ARMS, ids=lambda s: s.name)
def test_intervention_changes_output(rig, spec):
    """Every arm must actually bite on an image prompt.

    This is the check that catches a patch applied at a point the production
    path never reaches -- the failure mode that silently invalidated an
    earlier version of this study.
    """
    _, generate = rig
    base = [generate(i) for i in range(6)]
    got = [generate(i, spec) for i in range(6)]
    assert base != got, f"{spec.name} left every answer unchanged: the patch is not reached"


@pytest.mark.parametrize("spec", ALL_ARMS, ids=lambda s: s.name)
def test_restore_is_exact(rig, spec):
    """Leaving the context manager must return the model to baseline behaviour."""
    model, generate = rig
    base = [generate(i) for i in range(3)]
    with ablated(model, spec):
        pass
    assert [generate(i) for i in range(3)] == base
