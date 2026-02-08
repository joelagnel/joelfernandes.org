# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Joel Fernandes' personal technical blog (joelfernandes.org/linuxinternals.org) built with Jekyll. The site focuses on Linux kernel development, BPF/eBPF, tracing, and embedded systems content.

## Development Commands

### Prerequisites
```bash
apt-get install jekyll
gem install jekyll-paginate
```

### Local Development
```bash
jekyll serve
# Site will be available at http://localhost:4000
```

### Build and Deploy
```bash
./build-site.sh
# Builds site and pushes to GitHub Pages (gh-pages branch)
```

### Manual Build
```bash
jekyll build
# Output goes to _site/ directory
```

## Architecture

- **Source**: `master` branch contains Jekyll source files
- **Deployment**: `gh-pages` branch serves the built site on GitHub Pages
- **Build Process**: Manual build/deploy via `build-site.sh` (not automatic GitHub Pages)
- **URL**: http://www.linuxinternals.org

### Key Directories
- `_posts/`: Blog posts in Markdown with YAML frontmatter
- `_layouts/`: Page templates (post, page, bloglist, etc.)
- `_includes/`: Reusable components (header, footer, navigation, sidebars)
- `_plugins/`: Custom Jekyll plugins (image_tag, youtube, etc.)
- `resources/`: PDF presentations and technical documents
- `sass/`: SCSS stylesheets

### Content Structure
- Posts use filename format: `YYYY-MM-DD-title.markdown`
- Pagination: 1 post per page in `/blog/` directory
- Comments: Disqus integration (`linuxinternals1`)
- Sidebar: Configurable asides (delicious, pinboard, googleplus)

## Site Configuration
- Title: "JoelFernandes.org"
- Author: Joel Fernandes (GitHub: joelagnel)
- Plugins: jekyll-paginate
- Custom plugins in `_plugins/` directory