#!/bin/bash -e

# Build script for Hugo site
# Similar to the Jekyll build-site.sh but for Hugo

echo "Building Hugo site..."

# Clean any existing public directory
rm -rf public

# Build the Hugo site
hugo --cleanDestinationDir

echo "Hugo site built successfully in public/ directory"

# If you want to deploy to GitHub Pages, uncomment the following:
# git clone git@github.com:joelagnel/joelfernandes.org.git public -b gh-pages
# hugo
# cd public
# git add *
# git commit -asm "Update Hugo site on $(date)" --no-verify
# git push origin HEAD:gh-pages