COMPOSE_FILE=docker/docker-compose.yaml

up: ## Docker compose up
	docker compose -f $(COMPOSE_FILE) up -d
down: ## Docker compose down
	docker compose -f $(COMPOSE_FILE) down
build: ## Docker compose build
	docker compose -f $(COMPOSE_FILE) build
shell: ## Shell into container
	docker exec alba_api bash
prepare: ## Prepare the environment
	sudo mkdir -p volumes.prod
	sudo touch volumes.prod/db.sqlite



.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST)  | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
