#!/bin/bash -x

rm -rf _site
git clone git@github.com:joelagnel/joelfernandes.org.git _site -b gh-pages
jekyll build
cd _site
git add *
git commit -asm "Update on $(date)"
git push origin HEAD:gh-pages
