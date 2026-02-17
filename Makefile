#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = iris_project
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Remove data, models and logs artifacts
.PHONY: clean_artifacts
clean_artifacts:
	rm -rf data/bronze data/silver data/gold models logs

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format


## Run tests with pytest
.PHONY: test
test:
	uv run pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New venv created. Activate with: source .venv/bin/activate"


## Run bronze ingestion
.PHONY: bronze
bronze:
	uv run python data_processing/bronze/ingest_bronze.py

## Run silver cleaning
.PHONY: silver
silver:
	uv run python data_processing/silver/clean_data.py

## Run silver validation
.PHONY: validate
validate:
	uv run python data_processing/silver/validate_data.py

## Run feature pipeline
.PHONY: feature-pipeline
feature-pipeline:
	uv run python iris_project/pipelines/feature_pipeline.py

## Run training pipeline
.PHONY: training-pipeline
training-pipeline:
	uv run python iris_project/pipelines/training_pipeline.py

## Run full pipeline (bronze -> silver -> validate -> features -> training)
.PHONY: full-pipeline
full-pipeline: bronze silver validate feature-pipeline training-pipeline


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
