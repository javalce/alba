up: ## Docker compose up
	docker compose up -d --build
down: ## Docker compose down
	docker compose down
build: ## Docker compose build
	docker compose build
shell: ## Shell into container
	docker compose exec alba_api bash



.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST)  | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
