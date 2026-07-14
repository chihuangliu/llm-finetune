SHELL := /bin/bash

VENV := .venv-gguf
ACTIVATE := source $(VENV)/bin/activate
CONVERT_SCRIPT := third_party/llama.cpp/convert_hf_to_gguf.py
MODELFILE_TEMPLATE := templates/Modelfile

# Files that merge.py does not save but are needed for inference / conversion.
TOKENIZER_FILES := tokenizer.json tokenizer_config.json chat_template.jinja

# Override on the command line, e.g.
#   make merge BASE_MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct \
#              LORA_MODEL_PATH=sft_output/checkpoint-26 \
#              OUTPUT_PATH=sft_output/merged/checkpoint-26 GGUF=true
BASE_MODEL_PATH ?=
LORA_MODEL_PATH ?=
OUTPUT_PATH ?=
GGUF ?= false

.PHONY: merge
merge:
	@if [ -z "$(LORA_MODEL_PATH)" ] || [ -z "$(OUTPUT_PATH)" ]; then \
		echo "Usage: make merge BASE_MODEL_PATH=<base> LORA_MODEL_PATH=<lora> OUTPUT_PATH=<out> [GGUF=true]"; \
		exit 1; \
	fi
	@echo ">> Merging LoRA adapter into base model..."
	$(ACTIVATE) && python scirpts/merge.py \
		--base-model-path "$(BASE_MODEL_PATH)" \
		--lora-model-path "$(LORA_MODEL_PATH)" \
		--output-path "$(OUTPUT_PATH)"
	@echo ">> Copying tokenizer files into $(OUTPUT_PATH)..."
	@for f in $(TOKENIZER_FILES); do \
		if [ -f "$(LORA_MODEL_PATH)/$$f" ]; then \
			cp "$(LORA_MODEL_PATH)/$$f" "$(OUTPUT_PATH)/$$f"; \
			echo "   copied $$f"; \
		else \
			echo "   WARNING: $$f not found in $(LORA_MODEL_PATH), skipping"; \
		fi; \
	done
	@if [ "$(GGUF)" = "true" ]; then \
		$(MAKE) to-gguf MODEL_PATH="$(OUTPUT_PATH)"; \
	else \
		echo ">> GGUF=false, skipping GGUF conversion"; \
	fi

.PHONY: to-gguf
to-gguf:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Usage: make to-gguf MODEL_PATH=<hf-model-dir>"; \
		exit 1; \
	fi
	@echo ">> Converting $(MODEL_PATH) to GGUF (f16)..."
	$(ACTIVATE) && python $(CONVERT_SCRIPT) "$(MODEL_PATH)" \
		--outfile "$(MODEL_PATH)/model-f16.gguf" \
		--outtype f16
	@echo ">> GGUF written to $(MODEL_PATH)/model-f16.gguf"
	@$(MAKE) modelfile MODEL_PATH="$(MODEL_PATH)"

.PHONY: modelfile
modelfile:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Usage: make modelfile MODEL_PATH=<hf-model-dir>"; \
		exit 1; \
	fi
	@echo ">> Writing Modelfile into $(MODEL_PATH)..."
	@cp "$(MODELFILE_TEMPLATE)" "$(MODEL_PATH)/Modelfile"
	@echo ">> Modelfile written to $(MODEL_PATH)/Modelfile"
