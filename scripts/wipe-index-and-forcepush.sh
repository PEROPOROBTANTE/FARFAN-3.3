#!/usr/bin/env bash
set -euo pipefail

TARGET="index.html"
TIMESTAMP=$(date +%Y%m%dT%H%M%S)
REPO_ROOT="$(pwd)"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: ejecuta esto desde la raíz del repositorio git." >&2
  exit 1
fi

ORIGIN_URL=$(git config --get remote.origin.url || true)
if [ -z "$ORIGIN_URL" ]; then
  echo "ERROR: no hay remote 'origin' configurado. Añade el remote y vuelve a ejecutar." >&2
  exit 1
fi

PARENT_DIR="$(dirname "$REPO_ROOT")"
BACKUP_NAME="$(basename "$REPO_ROOT")-backup-$TIMESTAMP"
BACKUP_PATH="$PARENT_DIR/$BACKUP_NAME"
echo "Creando backup local en: $BACKUP_PATH"
cd "$PARENT_DIR"
cp -a "$REPO_ROOT" "$BACKUP_PATH"
cd "$REPO_ROOT"

# Asegurarse de que index.html no está en working tree
git rm -f --ignore-unmatch "$TARGET" || true
git commit -m "Remove $TARGET from working tree (preparing history rewrite)" || true

# Rama de respaldo
git branch -f backup-before-history-rewrite

# Reescribir historial para eliminar el archivo
echo "Reescribiendo historial para eliminar todas las apariciones de: $TARGET"
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch $TARGET" --prune-empty --tag-name-filter cat -- --all

# Limpiar referencias originales y compactar objetos
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Forzar push
echo "Forzando push al remoto (origin). Esto sobrescribirá el historial remoto."
git push origin --force --all
git push origin --force --tags

# Verificación
echo "Verificando: git log --all -- $TARGET (debe no mostrar commits)"
git log --all -- "$TARGET" || true

echo "Hecho. BACKUP guardado en: $BACKUP_PATH"
echo "IMPORTANTE: Todos los colaboradores deben re-clonar o resetear sus copias locales después de este force-push."
