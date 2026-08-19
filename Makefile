.DEFAULT_GOAL := help
SHELL := /bin/bash

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-16s %s\n", $$1, $$2}'

build: ## Build the image
	docker compose build

up: ## Start the service
	docker compose up -d

down: ## Stop the service
	docker compose down

logs: ## Follow the logs
	docker compose logs -f comfyui

shell: ## Interactive shell in the running container
	docker compose exec comfyui bash

verify: ## Run the full verification suite
	./scripts/verify-gpu.sh
	./scripts/verify-entrypoint.sh
	./scripts/verify-http.sh
	./scripts/verify-persistence.sh
	./scripts/verify-e2e.sh

reset-venv: ## Delete the overlay venv; recreated on next start
	@# .env is sourced with bash's own `.` here, scoped to this recipe only —
	@# not a blanket `-include .env` / `export` at file scope. Make's
	@# -include does not strip shell quoting, so a quoted value like
	@# COMFYUI_ARGS="..." would reach every recipe's environment (and then
	@# `docker compose`, via its env-precedence-over-.env rule) with the
	@# literal quote characters still attached, corrupting it. `set -a` /
	@# `set +a` exports only for the duration of this one shell invocation.
	@set -a; [ -f .env ] && . ./.env; set +a; \
	 echo "Removing $${COMFYUI_DATA_PATH:-/home/scott/LLMs/comfyui}/venv"; \
	 rm -rf "$${COMFYUI_DATA_PATH:-/home/scott/LLMs/comfyui}/venv"

update-comfyui: ## Print the newest upstream tag and SHA to pin
	@tag=$$(git ls-remote --tags --refs https://github.com/Comfy-Org/ComfyUI.git \
	        | sed 's#.*refs/tags/##' | sort -V | tail -1); \
	sha=$$(git ls-remote https://github.com/Comfy-Org/ComfyUI.git "refs/tags/$$tag" | cut -f1); \
	echo "newest tag: $$tag"; \
	echo "sha:        $$sha"; \
	echo "Paste into the Dockerfile ARG COMFYUI_REF, update the pinned date, then: make build verify"

fetch-model: ## Download a model: make fetch-model URL=... DEST=checkpoints
	./scripts/fetch-model.sh "$(URL)" "$(DEST)"

.PHONY: help build up down logs shell verify reset-venv update-comfyui fetch-model
