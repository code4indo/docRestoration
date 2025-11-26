#!/bin/bash
##############################################################################
# Stop Loghi Services
# Author: belekok
# Date: 2025-11-24
##############################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVICES_DIR="${SCRIPT_DIR}/loghi_services"

echo -e "${YELLOW}Stopping Loghi Services...${NC}"

cd "$SERVICES_DIR"

# Use docker compose or docker-compose
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

$DOCKER_COMPOSE down

echo -e "${GREEN}✓ Loghi Services stopped${NC}"
