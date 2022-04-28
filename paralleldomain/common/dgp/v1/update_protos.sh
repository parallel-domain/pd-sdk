#!/usr/bin/env bash

# Mac OSX users should to install `gnu-sed` so import renaming works
# https://stackoverflow.com/a/60562182/628496

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/dgp" || exit

find . -type d -exec rm "{}/__init__.py" \;

git pull git@github.com:TRI-ML/dgp.git master --force
git-filter-repo --path dgp/proto --path dgp/contribs/pd --path-rename dgp/proto:proto --path-rename dgp/contribs/pd:contribs/pd --force

find . -type f -iname "*_pb2.py" -exec sed -i 's/from dgp.proto/from paralleldomain.common.dgp.v1.dgp.proto/g' {} \;

find . -type d -exec touch "{}/__init__.py" \;

