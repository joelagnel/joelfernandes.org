# Hugo Port of JoelFernandes.org

This directory contains a Hugo port of the original Jekyll-based blog at joelfernandes.org.

## Overview

This Hugo site maintains the identical visual appearance and functionality of the original Jekyll site while providing:
- Faster build times
- Modern static site generator features
- Better performance
- Simplified deployment

## Prerequisites

Install Hugo:
```bash
# On Ubuntu/Debian
sudo apt-get install hugo

# On macOS
brew install hugo

# Or download from https://github.com/gohugoio/hugo/releases
```

## Development Commands

### Local Development
```bash
hugo server
# Site will be available at http://localhost:1313
```

### Build Site
```bash
./build.sh
# Output goes to public/ directory
```

### Manual Build
```bash
hugo --cleanDestinationDir
```

## Site Structure

- `content/` - Markdown content files (posts, pages)
- `layouts/` - Hugo templates (equivalent to Jekyll's _layouts)
- `static/` - Static assets (CSS, JS, images, PDFs)
- `hugo.toml` - Hugo configuration
- `build.sh` - Build script

## Content Management

### Adding Blog Posts
1. Create a new markdown file in `content/posts/`
2. Use the filename format: `YYYY-MM-DD-title.md`
3. Include frontmatter with title, date, and categories

### Static Pages
- Pages are in `content/` directories
- The `_index.html` file serves as the homepage
- Categories page is at `content/categories/index.html`
- Archive page is at `content/blog/archives/index.html`

## Configuration

The `hugo.toml` file contains:
- Site metadata (title, author, URL)
- Pagination settings
- Permalink structure matching the original Jekyll site
- Taxonomy configuration for categories

## Deployment

The site is configured to maintain the same URL structure as the Jekyll version:
- Blog posts: `/blog/YYYY/MM/DD/slug/`
- Categories: `/categories/`
- Archives: `/blog/archives/`

## Migration Notes

This Hugo port maintains complete visual and functional compatibility with the original Jekyll site:
- All CSS and JavaScript files are preserved
- Layouts are converted from Jekyll Liquid to Hugo templates
- Posts and pages maintain their original structure
- Categories and archives work identically
- All static assets (images, PDFs, resources) are included