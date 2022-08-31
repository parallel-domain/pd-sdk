#!/usr/bin/env bash

# Mac OSX users should to install `gnu-sed` so import renaming works
# https://stackoverflow.com/a/60562182/628496

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR" || exit
rm -rf src

git clone git@github.com:TRI-ML/dgp.git src
cd src || exit
git-filter-repo --path dgp/proto --path dgp/contribs/pd --force

# overwrite geometry.proto
cp ../geometry.proto dgp/proto/geometry.proto

# compile _pb2.py
protoc -I . --python_out=. dgp/proto/geometry.proto

# delete RPC files
find . -type f -iname "*_grpc.py" -exec rm {} \;

if [[ "$(uname)" == "Darwin" ]]; then # OS X uses different 'sed' arguments
  # disable Flake8 linting on pre-built python files
  find . -type f -iname "*_pb2.py" -exec sed -i '' -e '1s/^/# flake8: noqa \n/' {} \;
  # change relative to absolute imports
  find . -type f -iname "*_pb2.py" -exec sed -i '' -e 's/from dgp.proto/from paralleldomain.common.dgp.v1.src.dgp.proto/g' {} \;
else
  # disable Flake8 linting on pre-built python files
  find . -type f -iname "*_pb2.py" -exec sed -i '1s/^/# flake8: noqa \n/' {} \;
  # change relative to absolute imports
  find . -type f -iname "*_pb2.py" -exec sed -i 's/from dgp.proto/from paralleldomain.common.dgp.v1.src.dgp.proto/g' {} \;
fi



# make every folder a submodule
find . -type d -exec touch "{}/__init__.py" \;

rm -rf .git
