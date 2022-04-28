#!/usr/bin/env bash

# Mac OSX users should to install `gnu-sed` so import renaming works
# https://stackoverflow.com/a/60562182/628496

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR" || exit
rm -rf src

git clone git@github.com:TRI-ML/dgp.git src
cd src || exit
git-filter-repo --path dgp/proto --path dgp/contribs/pd --force

find . -type f -iname "*_pb2.py" -exec sed -i 's/from dgp.proto/from paralleldomain.common.dgp.v1.src.dgp.proto/g' {} \;
find . -type d -exec touch "{}/__init__.py" \;

rm -rf .git

