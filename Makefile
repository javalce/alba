COMPOSE_FILE=docker/docker-compose.yaml

up: ## Docker compose up
	docker compose -f $(COMPOSE_FILE) up -d
down: ## Docker compose down
	docker compose -f $(COMPOSE_FILE) down
build: ## Docker compose build
	docker compose -f $(COMPOSE_FILE) build
shell: ## Shell into container
	docker compose -f $(COMPOSE_FILE) exec alba_api bash
prepare: ## Prepare the environment
	@bash -c 'mkdir -p volumes.prod/{etcd,minio,milvus,ollama,logs}'
	@bash -c 'touch volumes.prod/{db.sqlite,logs/log.log}'
ollama: ## Initialize ollama model
	docker compose -f $(COMPOSE_FILE) exec -d ollama ollama run llama3.2
initdb: ## Initialize the database
	docker compose -f $(COMPOSE_FILE) exec -d alba_api alba db init
clean: ## Delete persistent data
	@read -p "Are you sure you want to delete these directories? [y/N] " confirm && \
	if [ "$$confirm" = "y" ]; then \
		sudo rm -rf volumes.prod; \
	else \
		echo "Abort"; \
	fi



.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST)  | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
