#!/bin/bash

set -e

rm -rf _site
jekyll build
mkdir -p _site/.git/
cd _site
git init
git add *
git commit -asm site
cd ..
cp git-config-gh-pages _site/.git/config
cd _site
git push origin HEAD:gh-pages --force
