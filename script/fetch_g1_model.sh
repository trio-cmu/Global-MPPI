#!/bin/bash
# Download the Unitree G1 MJCF model (about 35 MB, not vendored in this repo) into global_mppi/models/g1/.
# Source: https://github.com/vincekurtz/hydrax, hydrax/models/g1/, pinned to commit bd53d43 (the commit the results were produced with).
# The model is adopted by hydrax from MuJoCo Menagerie (unitree_g1), (c) Unitree Robotics, BSD-3-Clause; its LICENSE file is downloaded with it.
# The files are checked against script/g1_model.sha256. usage: script/fetch_g1_model.sh
set -e
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST=$REPO/global_mppi/models/g1
SHA=bd53d43d414e19557d38ee6c72613bf2e39fe834
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
git clone -q --filter=blob:none --no-checkout https://github.com/vincekurtz/hydrax "$TMP/hydrax"
git -C "$TMP/hydrax" sparse-checkout set hydrax/models/g1
git -C "$TMP/hydrax" checkout -q "$SHA"
rm -rf "$DEST"; mkdir -p "$DEST"
cp -r "$TMP/hydrax/hydrax/models/g1/." "$DEST/"
(cd "$DEST" && sha256sum --quiet -c "$REPO/script/g1_model.sha256") && echo "G1 model OK: $DEST ($(du -sh "$DEST" | cut -f1))"
