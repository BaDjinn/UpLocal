#! /bin/bash

(
  echo "===== TREE ====="
  tree -a -I "node_modules|dist|.git"
  echo
  echo "===== FILES ====="
  git ls-files | while read file; do
    echo
    echo "===== FILE: $file ====="
    sed 's/\t/    /g' "$file"
  done
) > uplocal_repo_dump.txt
