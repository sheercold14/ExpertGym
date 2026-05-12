#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${STORAGE_ROOT:-/tmp/shared-storage/ExpertGym/LLaMA}"
SOURCE_ROOT="${SOURCE_ROOT:-${STORAGE_ROOT}/sources}"
MRL_REPO="${MRL_REPO:-/mnt/cache/wuruixiao/users/lsc/mrl}"
BFCL_REPO="${BFCL_REPO:-/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard}"
ZEROSEARCH_REPO="${ZEROSEARCH_REPO:-${SOURCE_ROOT}/ZeroSearch}"

mkdir -p "${SOURCE_ROOT}"

if [ ! -d "${MRL_REPO}/.git" ]; then
  MRL_REPO="${SOURCE_ROOT}/mrl"
  if [ ! -d "${MRL_REPO}/.git" ]; then
    git clone https://github.com/xiangchi-yuan/mrl.git "${MRL_REPO}"
  fi
fi

if [ ! -d "${BFCL_REPO}/bfcl_eval" ]; then
  GORILLA_REPO="${SOURCE_ROOT}/gorilla"
  if [ ! -d "${GORILLA_REPO}/.git" ]; then
    git clone https://github.com/ShishirPatil/gorilla.git "${GORILLA_REPO}"
  fi
  BFCL_REPO="${GORILLA_REPO}/berkeley-function-call-leaderboard"
fi

if [ ! -d "${ZEROSEARCH_REPO}/.git" ]; then
  git clone https://github.com/Alibaba-NLP/ZeroSearch.git "${ZEROSEARCH_REPO}"
fi

for repo in "${MRL_REPO}" "${BFCL_REPO}" "${ZEROSEARCH_REPO}"; do
  if [ -d "${repo}/.git" ]; then
    printf '%s %s\n' "${repo}" "$(git -C "${repo}" rev-parse HEAD)"
  elif git -C "${repo}" rev-parse --show-toplevel >/dev/null 2>&1; then
    printf '%s %s\n' "${repo}" "$(git -C "${repo}" rev-parse HEAD)"
  else
    printf '%s\n' "${repo}"
  fi
done
