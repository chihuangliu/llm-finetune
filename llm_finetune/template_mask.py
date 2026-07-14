from transformers import AutoTokenizer
import re


class TemplateMaskError(Exception):
    pass


_QWEN_CHATML = (
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}",
    "{%- if message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{% generation %}{{- message.content }}{% endgeneration %}"
    "{{- '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- endif %}",
)

ASSISTANT_MASK_PATCHES = {
    "qwen-chatml": _QWEN_CHATML,
    # "llama3": (_LLAMA3_OLD, _LLAMA3_NEW),
}


def _is_mask_work(tokenizer: AutoTokenizer) -> bool:
    return (
        tokenizer.chat_template is not None
        and re.search(r"\{%-?\s*generation\s*-?%\}", tokenizer.chat_template)
        is not None
    )


def add_assistant_mask(tok, patches=ASSISTANT_MASK_PATCHES):

    if _is_mask_work(tok):
        return tok

    tpl = tok.chat_template
    for name, (old, new) in patches.items():
        if old in tpl:
            tok.chat_template = tpl.replace(old, new)
            if not _is_mask_work(tok):
                raise TemplateMaskError(
                    f"Patch '{name}' applied, but the assistant mask is still not working."
                )
            return tok

    raise TemplateMaskError("No Available template.")
