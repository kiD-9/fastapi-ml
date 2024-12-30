# Define targets
.PHONY: pretty test

pretty: isort black

black:
	black . --exclude env/

isort:
	isort . --skip env/

tests:
	pytest tests/integration_test.py tests/unit_test.py