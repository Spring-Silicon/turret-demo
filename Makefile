.PHONY: validate frontend-thor frontend-arc frontend-combined

FRONTEND_PORT ?= 8080
ARC_BACKEND ?= http://100.95.161.99:8080
THOR_BACKEND ?= http://100.88.90.48:8080

frontend-thor:
	node scripts/frontend.mjs --thor $(THOR_BACKEND) --port $(FRONTEND_PORT)

frontend-arc:
	node scripts/frontend.mjs --arc $(ARC_BACKEND) --port $(FRONTEND_PORT)

frontend-combined:
	node scripts/frontend.mjs --arc $(ARC_BACKEND) --thor $(THOR_BACKEND) --port $(FRONTEND_PORT)

validate:
	./scripts/validate.sh
