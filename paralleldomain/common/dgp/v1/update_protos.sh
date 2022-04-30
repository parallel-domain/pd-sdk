#!/usr/bin/env bash

# Mac OSX users should to install `gnu-sed` so import renaming works
# https://stackoverflow.com/a/60562182/628496

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR" || exit
rm -rf src

#git clone git@github.com:TRI-ML/dgp.git src
# Overwrite master TRI DGP repo. Use instead custom fork with distortion coefficients
# until https://github.com/TRI-ML/dgp/pull/86 is on master
git clone git@github.com:chrisochoatri/dgp.git src
cd src || exit
git checkout camera_distortion
git-filter-repo --path dgp/proto --path dgp/contribs/pd --force

# disable Flake8 linting on pre-built python files
find . -type f -iname "*_pb2.py" -exec sed -i '1s/^/# flake8: noqa \n/' {} \;
find . -type f -iname "*_pb2_grpc.py" -exec sed -i '1s/^/# flake8: noqa \n/' {} \;

# change relative to absolute imports
find . -type f -iname "*_pb2.py" -exec sed -i 's/from dgp.proto/from paralleldomain.common.dgp.v1.src.dgp.proto/g' {} \;

# make every folder a submodule
find . -type d -exec touch "{}/__init__.py" \;

rm -rf .git

