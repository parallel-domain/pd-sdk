#!/usr/bin/env bash

# Mac OSX users should to install `gnu-sed` so import renaming works
# https://stackoverflow.com/a/60562182/628496

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/src" || exit

find . -type d -exec rm "{}/__init__.py" \;

git stash
git clean -fd
git reset --hard
git pull git@github.com:TRI-ML/dgp.git master --rebase
git-filter-repo --path dgp/proto --path dgp/contribs/pd --force

find . -type f -iname "*_pb2.py" -exec sed -i 's/from dgp.proto/from paralleldomain.common.dgp.v1.src.dgp.proto/g' {} \;

find . -type d -exec touch "{}/__init__.py" \;

