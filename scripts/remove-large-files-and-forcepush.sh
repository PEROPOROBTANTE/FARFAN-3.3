#!/usr/bin/env bash
set -euo pipefail

# Archivo objetivo (ya lo dijiste): index.html
TARGET_NAME="index.html"
# Tamaño límite (GitHub = 100 MiB)
LIMIT_BYTES=$((100 * 1024 * 1024))

# comprobar que estamos en un repo
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: ejecuta esto desde la raíz del repositorio git." >&2
  exit 1
fi

ORIGIN_URL=$(git config --get remote.origin.url || true)
if [ -z "$ORIGIN_URL" ]; then
  echo "ERROR: no hay remote 'origin' configurado. Añade remote y vuelve a ejecutar." >&2
  exit 1
fi

# backup rápido (local)
TS=$(date +%Y%m%dT%H%M%S)
REPO_ROOT="$(pwd)"
PARENT_DIR="$(dirname "$REPO_ROOT")"
BACKUP_PATH="$PARENT_DIR/$(basename "$REPO_ROOT")-backup-$TS"
echo "Creando backup local: $BACKUP_PATH"
cd "$PARENT_DIR"
cp -a "$REPO_ROOT" "$BACKUP_PATH"
cd "$REPO_ROOT"

echo "Buscando blobs mayores a 100MB (esto puede tardar)..."
# Produce lines: <size> <sha> <path>
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1=="blob" {print $3, $2, substr($0, index($0,$4)) }' \
  | while read -r size sha path; do
      if [ "$size" -ge "$LIMIT_BYTES" ]; then
        printf "%s %s %s\n" "$size" "$sha" "$path"
      fi
    done > /tmp/large_blobs_$$.txt || true

if [ ! -s /tmp/large_blobs_$$.txt ]; then
  echo "No se encontraron blobs >100MB. Aún así limpiaré cualquier rastro de $TARGET_NAME si existe en historia."
else
  echo "Archivos/en blobs que superan 100MB (size sha path):"
  cat /tmp/large_blobs_$$.txt
fi

# Construir lista de paths grandes únicos
awk '{ $1=""; $2=""; sub(/^  /,""); print }' /tmp/large_blobs_$$.txt | sort -u > /tmp/large_paths_$$.txt || true

# Asegurarse de que si TARGET_NAME está en working tree, lo quitamos
git rm -f --ignore-unmatch "$TARGET_NAME" || true
git commit -m "Remove $TARGET_NAME from working tree (prepare history rewrite)" || true

# Crear rama de respaldo
git branch -f backup-before-history-rewrite

# Si BFG existe y hay remote, usarlo — más seguro/rápido
if command -v bfg >/dev/null 2>&1; then
  echo "BFG detectado: usaremos BFG para eliminar archivos grandes y $TARGET_NAME si hace falta."
  MIRROR_DIR="$(mktemp -d)/repo-mirror.git"
  git clone --mirror "$ORIGIN_URL" "$MIRROR_DIR"
  cd "$MIRROR_DIR"

  # eliminar por nombre específico TARGET_NAME
  echo "Borrando por nombre: $TARGET_NAME"
  bfg --delete-files "$TARGET_NAME" || true

  # si hay paths encontrados >100MB, eliminar por path (cada linea)
  if [ -s /tmp/large_paths_$$.txt ]; then
    echo "Borrando rutas grandes encontradas:"
    while read -r p; do
      # BFG expects paths or patterns; use --delete-files by basename and also --delete-folders if needed
      echo "  -> $p"
      bfg --delete-files "$(basename "$p")" || true
    done < /tmp/large_paths_$$.txt
  fi

  # limpieza final en mirror
  git reflog expire --expire=now --all || true
  git gc --prune=now --aggressive || true

  echo "Force-push del mirror al remoto (sobrescribirá historial remoto)..."
  git push --force || { echo "ERROR: push falló desde el mirror"; exit 1; }
  echo "Hecho con BFG. Regresa al repo original."
  cd "$REPO_ROOT"
else
  echo "BFG no instalado. Usando git filter-branch (fallback). Esto será más lento."
  # Crear una lista de paths a remover: incluir index.html siempre
  TMP_REMOVE="/tmp/remove_paths_$$.txt"
  echo "$TARGET_NAME" > "$TMP_REMOVE"
  if [ -s /tmp/large_paths_$$.txt ]; then
    # agregar cada path único (solo la ruta relativa)
    awk '{ print }' /tmp/large_paths_$$.txt >> "$TMP_REMOVE"
  fi
  # dedupe
  sort -u -o "$TMP_REMOVE" "$TMP_REMOVE"

  echo "Rutas que se eliminarán del historial:"
  cat "$TMP_REMOVE"

  # Construir index-filter command que remueva cada ruta
  INDEX_FILTER_CMD="git rm -r --cached --ignore-unmatch"
  while read -r rp; do
    # proteger espacios
    INDEX_FILTER_CMD="$INDEX_FILTER_CMD \"${rp//\"/\\\"}\""
  done < "$TMP_REMOVE"

  # Ejecutar filter-branch (nota: la expansión aquí requiere eval)
  eval git filter-branch --force --index-filter "$INDEX_FILTER_CMD" --prune-empty --tag-name-filter cat -- --all

  # limpieza
  rm -rf .git/refs/original/
  git reflog expire --expire=now --all || true
  git gc --prune=now --aggressive || true

  # push forzado
  echo "Forzando push al origin de todas las ramas y tags..."
  git push origin --force --all || { echo "ERROR: git push --force --all falló"; exit 1; }
  git push origin --force --tags || { echo "ERROR: git push --force --tags falló"; exit 1; }
fi

# Verificación
echo "Verificando que $TARGET_NAME no aparece en historia..."
if git log --all -- "$TARGET_NAME" | grep -q .; then
  echo "ATENCIÓN: $TARGET_NAME sigue apareciendo en el historial."
else
  echo "$TARGET_NAME eliminado del historial."
fi

echo "Listado final de blobs >100MB (si quedan):"
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1=="blob" {print $3, $2, substr($0, index($0,$4)) }' \
  | while read -r size sha path; do
      if [ "$size" -ge "$LIMIT_BYTES" ]; then
        printf "%s %s %s\n" "$size" "$sha" "$path"
      fi
    done || true

echo "BACKUP guardado en: $BACKUP_PATH"
echo "IMPORTANTE: tras este force-push TODOS los colaboradores deben re-clonar o resetear sus copias locales."
exit 0
