# joelfernandes.org

Personal website for Joel Fernandes, built with Jekyll.

## Design Credits

This website's design is inspired by [Brendan Gregg's website](https://www.brendangregg.com/), who kindly allowed reuse of some design ideas. Brendan's design was in turn inspired by the [Nokia Bell Labs Dennis M. Ritchie memorial page](https://www.nokia.com/bell-labs/about/dennis-m-ritchie/).

## Prerequisites

- Ruby (2.7+)
- Bundler (`gem install bundler`)

## Setup

```bash
# Install dependencies
bundle install
```

## Local Development

```bash
# Build and serve locally
bundle exec jekyll serve --port 4000

# Or just build without serving
bundle exec jekyll build
```

The site will be available at http://localhost:4000

## Deployment

The site is hosted on GitHub Pages from the `gh-pages` branch.

To deploy:

```bash
# Build the site
bundle exec jekyll build

# Copy _site contents to gh-pages branch root
# (Keep CNAME file for custom domain)
```

## Directory Structure

- `_posts/` - Blog posts in Markdown
- `_layouts/` - Page templates
- `_includes/` - Reusable HTML components
- `_config.yml` - Jekyll configuration
- `page.css` - Main stylesheet (Brendan Gregg-inspired)
- `backup-old/` - Previous site version (Octopress/Hugo)

## Adding a New Blog Post

Create a new file in `_posts/` with the naming convention:

```
YYYY-MM-DD-title-slug.md
```

With frontmatter:

```yaml
---
layout: post
title: "Your Post Title"
date: YYYY-MM-DD
categories: [category1, category2]
---

Your content here...
```
