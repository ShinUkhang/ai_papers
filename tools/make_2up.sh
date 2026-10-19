#!/bin/bash
# 쪽별 해설 md → 해설 PDF → 원문과 2쪽 모아찍기 PDF
# 사용법: bash tools/make_2up.sh <..._쪽별해설.md> [...]
set -e
TOOLS="$(cd "$(dirname "$0")" && pwd)"
for md in "$@"; do
  md="$(cd "$(dirname "$md")" && pwd)/$(basename "$md")"
  base="${md%_쪽별해설.md}"
  python3 -I "$TOOLS/md2pdf.py" --plain "$md"
  if [ -f "${base}_원문.pdf" ]; then
    swift "$TOOLS/side_by_side.swift" "${base}_원문.pdf" "${base}_쪽별해설.pdf" "${base}_모아찍기.pdf"
  else
    echo "원문 없음: ${base}_원문.pdf (해설 PDF만 생성)"
  fi
done
