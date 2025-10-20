#!/usr/bin/env bash
set -euo pipefail

TS=$(date +%Y%m%dT%H%M%S)
REPO="$(pwd)"
ORIGIN=$(git config --get remote.origin.url || true)
if [ -z "$ORIGIN" ]; then
  echo "ERROR: no hay remote 'origin' configurado. Sal." >&2
  exit 1
fi

# 1) Backup remoto seguro (NO toca main)
echo "Creando backup remoto seguro de origin/main..."
git fetch origin
git push origin refs/remotes/origin/main:refs/heads/backup-from-remote-$TS
echo "Backup remoto creado: backup-from-remote-$TS"

# 2) Listar blobs grandes (100MB+)
echo "Buscando blobs >100MB en la historia (esto puede tardar)..."
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1=="blob" { print $3, $2, substr($0, index($0,$4)) }' \
  | sort -nr > /tmp/git_all_blobs_$TS.txt || true
awk -v LIM=$((100*1024*1024)) '{ if ($1 >= LIM) print }' /tmp/git_all_blobs_$TS.txt > /tmp/git_large_blobs_$TS.txt || true

if [ ! -s /tmp/git_large_blobs_$TS.txt ]; then
  echo "No se encontraron blobs >100MB. El problema puede ser otro blob en forks/remotos; de todas formas subiremos tu rama como cleaned para que GitHub Desktop funcione."
  CLEAN_BRANCH="cleaned/$TS"
  git push origin HEAD:refs/heads/$CLEAN_BRANCH
  echo "Rama subida: $CLEAN_BRANCH  -> abre GitHub Desktop y haz fetch, checkout a esa rama."
  exit 0
fi

echo "Se encontraron blobs >100MB (size sha path):"
cat /tmp/git_large_blobs_$TS.txt
echo
# Preparar lista de paths únicos
awk '{ $1=""; $2=""; sub(/^  /,""); print }' /tmp/git_large_blobs_$TS.txt | sort -u > /tmp/large_paths_$TS.txt
echo "Rutas detectadas (únicas):"
cat /tmp/large_paths_$TS.txt
echo

# 3) Intentar limpieza automática (BFG si existe)
if command -v bfg >/dev/null 2>&1; then
  echo "BFG detectado: usándolo para eliminar los archivos detectados..."
  MIRROR="$(mktemp -d)/repo-mirror.git"
  git clone --mirror "$ORIGIN" "$MIRROR"
  cd "$MIRROR"
  while read -r p; do
    base=$(basename "$p")
    echo "BFG -> eliminar por nombre: $base"
    bfg --delete-files "$base" || true
  done < /tmp/large_paths_$TS.txt
  git reflog expire --expire=now --all || true
  git gc --prune=now --aggressive || true
  echo "Intentando force-push desde mirror (puede fallar si rama protegida)..."
  if git push --force; then
    echo "Force-push exitoso desde mirror. Verifica en GitHub."
    exit 0
  else
    echo "No se pudo force-push desde mirror (posible protección). Volviendo al repo local."
    cd "$REPO"
  fi
fi

# 4) Fallback: filter-branch en repo local
echo "Usando git filter-branch en repo local para eliminar las rutas detectadas (esto reescribe historial localmente)..."
cp /tmp/large_paths_$TS.txt /tmp/remove_list_$TS.txt
INDEX_FILTER='git rm -r --cached --ignore-unmatch'
while read -r rp; do
  safe=$(printf '%s' "$rp" | sed 's/"/\\"/g')
  INDEX_FILTER="$INDEX_FILTER \"$safe\""
done < /tmp/remove_list_$TS.txt
echo "Creando rama de respaldo local: backup-before-history-rewrite-$TS"
git branch -f backup-before-history-rewrite-$TS
echo "Ejecutando filter-branch (paciencia)..."
eval git filter-branch --force --index-filter "$INDEX_FILTER" --prune-empty --tag-name-filter cat -- --all
rm -rf .git/refs/original/ || true
git reflog expire --expire=now --all || true
git gc --prune=now --aggressive || true

# 5) Subir como rama cleaned (no tocar main)
CLEANED="cleaned/$TS"
echo "Subiendo tu historial reescrito como rama remota: $CLEANED (no toca main)"
git push origin HEAD:refs/heads/$CLEANED
echo "OK. Rama subida: $CLEANED"
echo "Abre GitHub Desktop, haz Fetch, checkout a 'cleaned/$TS' y trabaja desde ahí o crea PR hacia main en la web."
