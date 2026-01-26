#!/bin/bash
# =============================================================================
# Izwi Audio - Development Helper Script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Izwi Audio - Development Environment${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_usage() {
    echo -e "${GREEN}Usage:${NC} $0 <command>"
    echo ""
    echo -e "${GREEN}Commands:${NC}"
    echo "  up        Start the development environment"
    echo "  down      Stop the development environment"
    echo "  shell     Open a shell in the dev container"
    echo "  backend   Start the Rust backend with hot reload"
    echo "  frontend  Start the Vite dev server"
    echo "  logs      Show logs from all services"
    echo "  clean     Remove all containers and volumes"
    echo "  build     Build production Docker image"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo "  $0 up          # Start dev environment"
    echo "  $0 shell       # Open shell in container"
    echo "  $0 backend     # Run backend with hot reload"
}

case "${1:-help}" in
    up)
        print_header
        echo -e "${YELLOW}Starting development environment...${NC}"
        docker compose -f docker-compose.dev.yml up -d dev
        echo -e "${GREEN}✓ Development container started${NC}"
        echo ""
        echo -e "Run ${YELLOW}$0 shell${NC} to open a shell in the container"
        ;;
    
    down)
        print_header
        echo -e "${YELLOW}Stopping development environment...${NC}"
        docker compose -f docker-compose.dev.yml down
        echo -e "${GREEN}✓ Development environment stopped${NC}"
        ;;
    
    shell)
        print_header
        echo -e "${YELLOW}Opening shell in development container...${NC}"
        docker compose -f docker-compose.dev.yml exec dev bash
        ;;
    
    backend)
        print_header
        echo -e "${YELLOW}Starting Rust backend with hot reload...${NC}"
        docker compose -f docker-compose.dev.yml exec dev \
            bash -c "export PATH=/app/.venv/bin:\$PATH && cargo watch -x run"
        ;;
    
    frontend)
        print_header
        echo -e "${YELLOW}Starting Vite dev server...${NC}"
        docker compose -f docker-compose.dev.yml exec dev \
            bash -c "cd ui && npm run dev -- --host"
        ;;
    
    logs)
        docker compose -f docker-compose.dev.yml logs -f
        ;;
    
    clean)
        print_header
        echo -e "${RED}Warning: This will remove all containers and volumes!${NC}"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker compose -f docker-compose.dev.yml down -v --remove-orphans
            docker compose down -v --remove-orphans 2>/dev/null || true
            echo -e "${GREEN}✓ Cleaned up${NC}"
        else
            echo "Cancelled"
        fi
        ;;
    
    build)
        print_header
        echo -e "${YELLOW}Building production Docker image...${NC}"
        docker compose build izwi
        echo -e "${GREEN}✓ Production image built${NC}"
        ;;
    
    build-cuda)
        print_header
        echo -e "${YELLOW}Building CUDA Docker image...${NC}"
        docker compose build izwi-cuda
        echo -e "${GREEN}✓ CUDA image built${NC}"
        ;;
    
    help|--help|-h|*)
        print_header
        print_usage
        ;;
esac
