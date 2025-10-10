#!/bin/bash
# Quick Start - Flexible Training with Docker Compose

cat <<'EOF'

╔══════════════════════════════════════════════════════════╗
║     🚀 GAN-HTR Flexible Training Configuration          ║
╚══════════════════════════════════════════════════════════╝

Pilihan training yang tersedia:

┌──────────────────────────────────────────────────────────┐
│ 1. 🧪 SMOKE TEST (Recommended untuk testing)             │
│    Quick validation - 2 epochs                           │
│    $ docker-compose up -d gan-htr-smoke-test             │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ 2. 🚀 PRODUCTION TRAINING (Full training)                │
│    Default: train.py dengan mixed precision              │
│    $ docker-compose up -d gan-htr-prod                   │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ 3. 🔧 CUSTOM SCRIPT (Via environment variable)           │
│    Run any training script flexibly                      │
│    $ TRAINING_SCRIPT=scripts/train32_smoke_test.sh \     │
│      docker-compose up -d gan-htr-prod                   │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ 4. 💻 DEVELOPMENT MODE (Interactive)                     │
│    Shell access untuk debugging                          │
│    $ docker-compose up -d gan-htr-dev                    │
│    $ docker exec -it gan-htr-dev bash                    │
└──────────────────────────────────────────────────────────┘

════════════════════════════════════════════════════════════

📋 QUICK EXAMPLES:

  Smoke Test:
  $ docker-compose up -d gan-htr-smoke-test
  $ docker logs -f gan-htr-smoke-test

  Full Training:
  $ docker-compose up -d gan-htr-prod
  $ docker logs -f gan-htr-prod

  Custom Script:
  $ TRAINING_SCRIPT=scripts/my_script.sh \
    docker-compose up -d gan-htr-prod

  Stop Training:
  $ docker-compose stop gan-htr-smoke-test

════════════════════════════════════════════════════════════

📚 Documentation:
  • FLEXIBLE_TRAINING_CONFIG.md - Full guide
  • CLOUD_DEPLOYMENT.md - Cloud deployment
  • .env.example - Environment variables

🔍 Status Check:
  $ docker-compose ps
  $ docker logs <container-name>

════════════════════════════════════════════════════════════

EOF
