#!/bin/bash
##############################################################################
# Check Loghi Services Status
# Author: belekok
# Date: 2025-11-24
##############################################################################

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Loghi Services Status Check${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Check Docker containers
echo -e "${YELLOW}Docker Containers:${NC}"
docker ps --filter "name=gan-htr-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "No containers running"
echo ""

# Check service endpoints
echo -e "${YELLOW}Service Health Check:${NC}"

# LAYPA
echo -n "LAYPA (port 5000): "
if curl -s --connect-timeout 2 http://localhost:5000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Not responding${NC}"
fi

# HTR
echo -n "HTR (port 5001): "
if curl -s --connect-timeout 2 http://localhost:5001 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Not responding${NC}"
fi

# TOOLING
echo -n "TOOLING (port 8082): "
if curl -s --connect-timeout 2 http://localhost:8082/api/info > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Not responding${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
