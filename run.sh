#!/usr/bin/env bash
dir=$(python -m site --user-site)
rm -rf "${dir}/xclib"
cp --verbose -rf xclib "${dir}/xclib"