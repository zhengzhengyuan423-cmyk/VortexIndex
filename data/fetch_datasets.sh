# 下载脚本
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

URL="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
[ -f siftsmall/siftsmall_base.fvecs ] && { echo "already present"; exit 0; }

for _ in $(seq 1 40); do
  if tar -tzf siftsmall.tar.gz >/dev/null 2>&1; then break; fi
  curl -s -C - --retry 3 --retry-delay 3 --max-time 280 -o siftsmall.tar.gz "$URL" || true
  sleep 3
done

tar -tzf siftsmall.tar.gz >/dev/null 2>&1 || { echo "download incomplete"; exit 1; }
tar -xzf siftsmall.tar.gz
ls -l siftsmall/
