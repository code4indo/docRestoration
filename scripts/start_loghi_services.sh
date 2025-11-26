#!/bin/bash
##############################################################################
# Start Loghi Services (LAYPA + HTR + TOOLING)
# Author: belekok
# Date: 2025-11-24
#
# This script starts all required Loghi services for document processing:
# - LAYPA: Layout analysis and baseline detection (port 5000)
# - HTR: Handwritten text recognition (port 5001)
# - TOOLING: Baseline extraction and XML processing (port 8080)
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

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Starting Loghi Services for GAN-HTR Integration${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker not found. Please install Docker first.${NC}"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}ERROR: Docker Compose not found.${NC}"
    exit 1
fi

# Use docker compose or docker-compose
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Check if services directory exists
if [ ! -d "$SERVICES_DIR" ]; then
    echo -e "${RED}ERROR: Services directory not found: $SERVICES_DIR${NC}"
    exit 1
fi

cd "$SERVICES_DIR"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}ERROR: .env file not found in $SERVICES_DIR${NC}"
    echo -e "${YELLOW}Please create .env file first.${NC}"
    exit 1
fi

# Load environment variables to validate paths
source .env

# Validate model paths
echo -e "${YELLOW}[1/5] Validating configuration...${NC}"
if [ ! -d "$LAYPA_MODEL_PATH" ]; then
    echo -e "${RED}ERROR: LAYPA model not found: $LAYPA_MODEL_PATH${NC}"
    exit 1
fi

if [ ! -d "$LOGHI_MODEL_PATH" ]; then
    echo -e "${RED}ERROR: HTR model not found: $LOGHI_MODEL_PATH${NC}"
    exit 1
fi

if [ ! -f "$TOOLING_CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: Tooling config not found: $TOOLING_CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Configuration validated${NC}"

# Create output directories if they don't exist
echo -e "${YELLOW}[2/5] Preparing output directories...${NC}"
mkdir -p "$LAYPA_OUTPUT_PATH" "$LOGHI_OUTPUT_PATH" "$TOOLING_OUTPUT_PATH"
echo -e "${GREEN}✓ Output directories ready${NC}"

# Check if services are already running
echo -e "${YELLOW}[3/5] Checking existing containers...${NC}"
if docker ps | grep -q "gan-htr-laypa\|gan-htr-htr\|gan-htr-tooling"; then
    echo -e "${YELLOW}⚠ Some services are already running${NC}"
    read -p "Stop and restart them? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Stopping existing containers...${NC}"
        $DOCKER_COMPOSE down
    else
        echo -e "${YELLOW}Keeping existing containers${NC}"
        exit 0
    fi
fi

# Start services
echo -e "${YELLOW}[4/5] Starting Docker services...${NC}"
echo -e "${BLUE}This may take a moment...${NC}"
$DOCKER_COMPOSE up -d

# Wait for services to be ready
echo -e "${YELLOW}[5/5] Waiting for services to be ready...${NC}"
sleep 5

# Health check
echo ""
echo -e "${BLUE}Checking service health...${NC}"

# Check LAYPA
if curl -s --connect-timeout 5 http://localhost:5000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ LAYPA is running (http://localhost:5000)${NC}"
else
    echo -e "${RED}✗ LAYPA failed to start${NC}"
fi

# Check HTR
if curl -s --connect-timeout 5 http://localhost:5001 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ HTR is running (http://localhost:5001)${NC}"
else
    echo -e "${RED}✗ HTR failed to start (may take longer to initialize)${NC}"
fi

# Check TOOLING
if curl -s --connect-timeout 5 http://localhost:8082/api/info > /dev/null 2>&1; then
    echo -e "${GREEN}✓ TOOLING is running (http://localhost:8082)${NC}"
else
    echo -e "${RED}✗ TOOLING failed to start (may take longer to initialize)${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Loghi Services Started Successfully!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Service URLs:"
echo -e "  • LAYPA:   ${BLUE}http://localhost:5000${NC} (external service)"
echo -e "  • HTR:     ${BLUE}http://localhost:5001${NC}"
echo -e "  • TOOLING: ${BLUE}http://localhost:8082${NC}"
echo ""
echo -e "Commands:"
echo -e "  • View logs:   ${YELLOW}$DOCKER_COMPOSE logs -f${NC}"
echo -e "  • Stop services: ${YELLOW}./stop_loghi_services.sh${NC}"
echo -e "  • Restart:     ${YELLOW}$DOCKER_COMPOSE restart${NC}"
echo ""
echo -e "${YELLOW}Note: HTR and TOOLING may take 10-30 seconds to fully initialize.${NC}"
echo -e "${YELLOW}      Check logs if services don't respond: cd $SERVICES_DIR && $DOCKER_COMPOSE logs${NC}"
echo ""
