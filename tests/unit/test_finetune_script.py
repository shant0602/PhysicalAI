from __future__ import annotations

import ast
from pathlib import Path

FINETUNE_PY = Path(__file__).parents[2] / "scripts" / "finetune.py"
TRAIN_SH = Path(__file__).parents[2] / "scripts" / "train.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_finetune() -> ast.Module:
    return ast.parse(FINETUNE_PY.read_text())


def _call_name(node: ast.Call) -> str:
    """Return dotted name of a Call node's function, e.g. 'torch.nn.utils.clip_grad_norm_'."""
    func = node.func
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def _find_calls(tree: ast.Module, name_suffix: str) -> list[int]:
    """Return sorted list of line numbers for all Call nodes whose name ends with name_suffix."""
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node).endswith(name_suffix)
    )


# ---------------------------------------------------------------------------
# Gradient clipping regression tests
# ---------------------------------------------------------------------------

def test_finetune_py_exists():
    assert FINETUNE_PY.exists(), f"scripts/finetune.py not found at {FINETUNE_PY}"


def test_finetune_py_parses():
    """Catches syntax errors introduced by future edits."""
    _parse_finetune()  # raises SyntaxError if broken


def test_clip_grad_norm_present():
    """clip_grad_norm_ must exist — its absence is the root cause of the mode collapse bug."""
    tree = _parse_finetune()
    lines = _find_calls(tree, "clip_grad_norm_")
    assert lines, (
        "clip_grad_norm_ call is missing from scripts/finetune.py — gradient clipping removed!"
    )


def test_clip_grad_norm_before_optimizer_step():
    """clip_grad_norm_ must appear before optimizer.step() — order matters for correctness."""
    tree = _parse_finetune()
    clip_lines = _find_calls(tree, "clip_grad_norm_")
    step_lines = _find_calls(tree, "optimizer.step")

    assert clip_lines, "clip_grad_norm_ call missing"
    assert step_lines, "optimizer.step call missing"

    # Every optimizer.step must be preceded by at least one clip_grad_norm_ earlier in the file
    first_clip = clip_lines[0]
    first_step = step_lines[0]
    assert first_clip < first_step, (
        f"clip_grad_norm_ (line {first_clip}) must come before optimizer.step (line {first_step})"
    )


def test_clip_grad_norm_max_norm_is_1():
    """max_norm=1.0 is the standard safe value — catch accidental changes."""
    tree = _parse_finetune()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _call_name(node).endswith("clip_grad_norm_")):
            continue
        for kw in node.keywords:
            if kw.arg == "max_norm":
                assert isinstance(kw.value, ast.Constant) and kw.value.value == 1.0, (
                    f"clip_grad_norm_ max_norm should be 1.0, got {ast.unparse(kw.value)}"
                )
                return
    raise AssertionError("clip_grad_norm_ called without explicit max_norm keyword")


# ---------------------------------------------------------------------------
# FinetuneConfig field regression tests
# ---------------------------------------------------------------------------

def test_finetune_config_has_lora_fields():
    """Ensure LoRA config fields are not accidentally removed."""
    tree = _parse_finetune()
    config_class = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "FinetuneConfig"),
        None,
    )
    assert config_class is not None, "FinetuneConfig class not found in finetune.py"

    field_names = {
        node.target.id
        for node in ast.walk(config_class)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    required = {"use_lora", "lora_rank", "lora_dropout", "use_quantization", "learning_rate"}
    missing = required - field_names
    assert not missing, f"FinetuneConfig is missing fields: {missing}"


# ---------------------------------------------------------------------------
# train.sh wiring test
# ---------------------------------------------------------------------------

def test_train_sh_uses_local_finetune_not_submodule():
    """train.sh must point to scripts/finetune.py, not the submodule copy."""
    content = TRAIN_SH.read_text()
    assert 'FINETUNE_SCRIPT="$REPO_ROOT/scripts/finetune.py"' in content, (
        "train.sh FINETUNE_SCRIPT must point to scripts/finetune.py (not the submodule copy)"
    )
    assert "third_party" not in content.split("FINETUNE_SCRIPT=")[1].split("\n")[0], (
        "train.sh FINETUNE_SCRIPT must not point into third_party/"
    )
