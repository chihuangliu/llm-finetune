"""Tests for llm_finetune.template_mask.

Two layers:
  - Real Qwen tokenizer: prove the chat_template is actually patched, that the
    rendered text is byte-identical (the model sees no difference), and that the
    assistant-only mask really selects only the assistant tokens.
  - FakeTokenizer unit tests: cover the registry / error branches without the
    heavy (network-dependent) tokenizer download.
"""

import pytest

from llm_finetune.template_mask import (
    ASSISTANT_MASK_PATCHES,
    TemplateMaskError,
    _is_mask_work,
    add_assistant_mask,
)

QWEN_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

CONVERSATION = [
    {"role": "system", "content": "You are Alex, a coding interviewer."},
    {"role": "user", "content": "Problem: Climbing Stairs ... passed=0/3"},
    {"role": "assistant", "content": "You haven't written any code yet."},
]


# --------------------------------------------------------------------------- #
# Real Qwen tokenizer                                                          #
# --------------------------------------------------------------------------- #
@pytest.fixture
def qwen_tok():
    """A fresh Qwen tokenizer per test (add_assistant_mask mutates it in place).

    Skips if the tokenizer can't be loaded (e.g. no cache / no network) so the
    suite still runs offline.
    """
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(QWEN_MODEL)
    except Exception as exc:  # network / cache miss
        pytest.skip(f"Qwen tokenizer unavailable: {exc}")


def test_qwen_starts_without_working_mask(qwen_tok):
    """Baseline: the stock Qwen template has no {% generation %} tags."""
    assert "{% generation %}" not in qwen_tok.chat_template
    assert _is_mask_work(qwen_tok) is False


def test_qwen_template_is_actually_patched(qwen_tok):
    original = qwen_tok.chat_template

    returned = add_assistant_mask(qwen_tok)

    assert returned is qwen_tok  # mutates in place and returns it
    assert qwen_tok.chat_template != original  # template really changed
    assert "{% generation %}" in qwen_tok.chat_template
    assert "{% endgeneration %}" in qwen_tok.chat_template
    assert _is_mask_work(qwen_tok) is True


def test_qwen_render_is_byte_identical(qwen_tok):
    """The patch must not change what the model sees: rendered text is unchanged."""
    original = qwen_tok.chat_template
    before = qwen_tok.apply_chat_template(
        CONVERSATION, tokenize=False, chat_template=original
    )

    add_assistant_mask(qwen_tok)
    after = qwen_tok.apply_chat_template(CONVERSATION, tokenize=False)

    assert before == after


def test_qwen_mask_selects_only_the_assistant(qwen_tok):
    add_assistant_mask(qwen_tok)

    out = qwen_tok.apply_chat_template(
        CONVERSATION, return_assistant_tokens_mask=True, return_dict=True
    )
    ids, mask = out["input_ids"], out["assistant_masks"]

    assert sum(mask) > 0
    assert sum(mask) < len(mask)  # not everything is unmasked

    kept = qwen_tok.decode([i for i, m in zip(ids, mask) if m == 1])
    assert kept.strip() == CONVERSATION[-1]["content"]
    # the system / user text must be excluded from the loss
    assert "coding interviewer" not in kept
    assert "Climbing Stairs" not in kept


def test_qwen_idempotent(qwen_tok):
    """Calling twice is a no-op the second time (mask already works)."""
    add_assistant_mask(qwen_tok)
    once = qwen_tok.chat_template

    add_assistant_mask(qwen_tok)  # early-returns via _is_mask_work

    assert qwen_tok.chat_template == once
    assert once.count("{% generation %}") == 1  # not double-patched


# --------------------------------------------------------------------------- #
# Registry / error branches (no download)                                     #
# --------------------------------------------------------------------------- #
class FakeTokenizer:
    """Minimal stand-in: the mask "works" iff a marker sits in the template."""

    MARKER = "{% generation %}"

    def __init__(self, chat_template):
        self.chat_template = chat_template

    def apply_chat_template(
        self, messages, return_assistant_tokens_mask=False, return_dict=False, **kw
    ):
        has_mask = self.MARKER in self.chat_template
        if return_dict:
            return {"input_ids": [0, 1], "assistant_masks": [1, 1] if has_mask else [0, 0]}
        return "rendered"


def test_already_working_returns_unchanged():
    tok = FakeTokenizer("prefix {% generation %} suffix")
    before = tok.chat_template

    result = add_assistant_mask(tok)

    assert result is tok
    assert tok.chat_template == before  # untouched, no patch applied


def test_no_matching_patch_raises():
    tok = FakeTokenizer("a template that matches no registered patch")

    with pytest.raises(TemplateMaskError, match="No Available template"):
        add_assistant_mask(tok)


def test_patch_matched_but_mask_still_broken_raises():
    """`old` matches so a patch is applied, but `new` adds no marker -> raise."""
    tok = FakeTokenizer("OLD_MARKER stays broken")
    broken_patches = {"broken": ("OLD_MARKER", "NEW_still_no_generation_tag")}

    with pytest.raises(TemplateMaskError, match="still not working"):
        add_assistant_mask(tok, patches=broken_patches)

    assert tok.chat_template == "NEW_still_no_generation_tag stays broken"


def test_registry_ships_qwen_patch():
    assert "qwen-chatml" in ASSISTANT_MASK_PATCHES
    old, new = ASSISTANT_MASK_PATCHES["qwen-chatml"]
    assert "{% generation %}" not in old
    assert "{% generation %}" in new
