# zty2004.github.io

Personal homepage & blog of **Tianyou Zuo (fztym)** — built with [Jekyll](https://jekyllrb.com/) on GitHub Pages.

🔗 **Live site:** <https://zty2004.github.io>

## Features

- 🌅 **Monet sunset theme** — hand-crafted palette inspired by *Impression, Sunrise*, with drifting light-ray background, canvas grain texture, and a cursive gradient signature
- 🌗 **Light / dark mode** — sun–moon slider fixed at the top-right, follows system preference, remembered via `localStorage`
- 🎓 **Academic homepage as the landing page** — profile, news, research interests and education timeline at `/`; the post list lives at `/blogs/`
- 🏷️ **Tags** — multi-select filtering (AND / OR) with shareable URLs at `/tags/`
- 🔍 **Search** — client-side full-text search at `/search/`, index fetched on demand, CJK-friendly
- 🖼️ **Photo galleries** — consecutive photos become a grid; every image in a post opens in a lightbox with ←/→ navigation and swipe support
- 🧮 **LaTeX math** — MathJax 3 loaded on demand via `math: true` front matter (always write formulas as `$$...$$`)
- 📊 **Mermaid diagrams** — flowcharts and sequence diagrams via `mermaid: true`, re-rendered on theme switch
- 🌲 **Collapsible TOC** — h2–h4 tree with `tree`-style guides, scroll tracking, desktop sidebar and mobile fold-out
- 💻 **Code blocks** — Monokai highlighting with copy buttons (always visible on touch devices)
- ⌨️ **Keyboard shortcuts** — `/` search, `g h/b/t/s` navigation, `j/k` headings, `t` theme, `c` TOC, `?` help
- 📈 **Reading progress bar** — sunset gradient line tracking position within the article
- 💬 **Comments & stats** — [giscus](https://giscus.app) (GitHub Discussions) with live theme sync, busuanzi visitor counters
- ⚡ **Performance** — WebP with JPEG fallback (219 MB → 87 MB), explicit image dimensions (no layout shift), index thumbnails, self-hosted subset webfont (3 KB), third-party scripts deferred until needed
- 🔎 **SEO** — `jekyll-seo-tag`, `BlogPosting` JSON-LD, thumbnail-based Open Graph cards, RSS with absolute image URLs

## Site map

| URL | Page |
|---|---|
| `/` | Academic homepage (profile, news, research, education) |
| `/blogs/` | Blog post list |
| `/tags/` | Tag cloud with multi-select filtering |
| `/search/` | Full-text search |
| `/<post-title>/` | Individual posts |

## Repository layout

```
├── _config.yml       # site-wide configuration
├── _pages/           # standalone pages: home (about.md), blogs, tags, search, 404
├── _posts/           # blog posts (YYYY-M-D-Title.md)
├── _layouts/         # default / page / post templates
├── _includes/        # meta, schema, icons, analytics, giscus snippets
├── _sass/            # theme partials (variables, reset, Monokai highlights, icons)
├── assets/
│   ├── css/          # main stylesheet entry (style.scss)
│   ├── js/           # TOC + copy buttons, gallery lightbox, shortcuts, mermaid
│   └── fonts/        # self-hosted subset cursive font (OFL)
├── images/           # post photos (JPEG + WebP siblings, web-optimised)
├── scripts/          # maintenance scripts (image optimiser, WebP, thumbs, font subset)
├── search.json       # search index (Liquid-generated)
├── feed.xml          # RSS feed with absolute image URLs
└── src/              # postgen CLI tool (C++20)
```

## Writing posts with `postgen`

A one-command blog post generator that handles naming, front matter, images, and publishing:

```bash
cd src && make                      # build once (requires clang++, C++20)

./src/bin/postgen note.md --tags notes            # markdown note
./src/bin/postgen paper.tex                       # LaTeX (needs pandoc), math auto-enabled
./src/bin/postgen slides.pdf --title "My Talk"    # embedded PDF post
./src/bin/postgen --photos ~/Pictures/Trip        # photo-album post
```

- Title is auto-extracted (front matter / `# H1` / `\title{}` / PDF metadata), or pass `--title`
- Local images are copied into `images/<date>-<slug>/` and compressed with `sips`
- Add `--publish` to git add + commit + push in one go
- Other flags: `--date`, `--math`, `--force`, `--dry-run`

## Local development

```bash
gem install jekyll bundler
jekyll serve          # http://localhost:4000
```

Pushing to `master` triggers the GitHub Pages build automatically.

## Credits & license

Originally scaffolded from [Jekyll Now](https://github.com/barryclark/jekyll-now) by Barry Clark (MIT), since heavily customized. Released under the [MIT License](LICENSE).
