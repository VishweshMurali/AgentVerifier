#!/usr/bin/env bash
# Run on the Ubuntu 24.04 Arm VM as the default user (ubuntu):  bash setup.sh
# Installs Docker, opens ports 80/443 on the machine's firewall, gets the code, and starts the app.
set -euo pipefail

BRANCH="${BRANCH:-public-site}"
REPO="${REPO:-https://github.com/VishweshMurali/AgentVerifier.git}"

echo "== Docker"
if ! command -v docker >/dev/null; then
  sudo apt-get update -y
  sudo apt-get install -y docker.io docker-compose-v2 git
  sudo systemctl enable --now docker
  sudo usermod -aG docker "$USER"
fi

echo "== Firewall (Oracle's Ubuntu image blocks everything but ssh by default)"
for port in 80 443; do
  sudo iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null ||
    sudo iptables -I INPUT 1 -p tcp --dport "$port" -j ACCEPT
done
sudo apt-get install -y iptables-persistent >/dev/null 2>&1 || true
sudo netfilter-persistent save >/dev/null 2>&1 || true

echo "== Code ($BRANCH)"
if [ -d "$HOME/AgentVerifier/.git" ]; then
  git -C "$HOME/AgentVerifier" fetch origin "$BRANCH"
  git -C "$HOME/AgentVerifier" checkout "$BRANCH"
  git -C "$HOME/AgentVerifier" pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO" "$HOME/AgentVerifier"
fi

cd "$HOME/AgentVerifier/deploy/oracle"
if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
  echo
  echo "Now type your proxy token into .env on THIS machine (never in chat):"
  echo "    nano $HOME/AgentVerifier/deploy/oracle/.env      (set GROQ_PROXY_TOKEN=...)"
  echo "then run:  cd $HOME/AgentVerifier/deploy/oracle && sudo docker compose up -d --build"
  exit 0
fi

echo "== Build and start (the first build takes several minutes)"
sudo docker compose up -d --build
sudo docker compose ps
