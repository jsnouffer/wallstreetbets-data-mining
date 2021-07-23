# HELP
.PHONY: help


help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

clean: ## Removes build artifacts and any logs
	rm -rf *.egg-info && rm -rf dist && rm -rf *.log* && rm -rf build
	rm -rf venv && rm -f Pipfile && rm -f Pipfile.lock

venv: ## Creates or updates project's virtual enviornment. To activate, run: source venv/bin/activate
	python3 -m venv venv --prompt venv
	venv/bin/pip install -r requirements.txt
	venv/bin/pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

update: ## Updates dependencies in virtual environment
	venv/bin/pip install -r requirements.txt
	venv/bin/pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

run: venv ## Launches app using virtual environment
	CONFIG_ENV=config.env venv/bin/python -m mbti
