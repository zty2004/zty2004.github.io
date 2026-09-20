# zty2004.github.io

Personal homepage & blog of **Tianyou Zuo (fztym)** — built with [Jekyll](https://jekyllrb.com/) on GitHub Pages.

🔗 **Live site:** <https://zty2004.github.io>

## Features

- 🌅 **Monet sunset theme** — hand-crafted palette inspired by *Impression, Sunrise*, with drifting light-ray background, canvas grain texture, and a cursive gradient signature
- 🌗 **Light / dark mode** — sun–moon slider fixed at the top-right, follows system preference, remembered via `localStorage`
- 🎓 **Academic homepage as the landing page** — profile, news, research interests and education timeline at `/`; the post list lives at `/blogs/`
- 🏷️ **Tags** — multi-select filtering (AND / OR) with shareable URLs at `/tags/`
- 🔍 **Search** — client-side full-text search at `/search/`, index fetched on demand, CJK-friendly
- 🖼️ **Adaptive photo layouts** — photo-only posts become a loosely hand-laid wall (real aspect ratios, gentle tilt) with Photos.app-style pinch / ⌘-wheel zoom from 1 to 8 across; posts that mix prose and photos pack each run of photos into tidy justified rows. Every image opens in a lightbox with ←/→ navigation and swipe support
- 🧮 **LaTeX math** — MathJax 3 loaded on demand via `math: true` front matter (`$...$` inline, `$$ ... $$` block for display)
- 📊 **Mermaid diagrams** — flowcharts and sequence diagrams via `mermaid: true`, re-rendered on theme switch
- 🌲 **Collapsible TOC** — h2–h4 tree with `tree`-style guides, scroll tracking, desktop sidebar and mobile fold-out
- 💻 **Code blocks** — Monokai highlighting with copy buttons (always visible on touch devices)
- ⌨️ **Keyboard shortcuts** — `/` search, `g h/b/t/s` navigation, `j/k` headings, `t` theme, `c` TOC, `?` help
- 📈 **Reading progress bar** — sunset gradient line tracking position within the article
- 💬 **Comments & stats** — [giscus](https://giscus.app) (GitHub Discussions) with live theme sync, busuanzi visitor counters
- ⚡ **Performance** — responsive WebP tiers (Japan post: 12 MB → 1.2 MB at 3-across on a 1× display, 3.2 MB on a 2× one, 0.48 MB zoomed out to 8-across), explicit image dimensions (no layout shift), index thumbnails, self-hosted subset webfont (3 KB), third-party scripts deferred until needed
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
├── images/           # post photos (JPEG + WebP in 200/480/800/1600 tiers)
├── scripts/          # image pipeline (optimise, WebP tiers, responsive markup, thumbs)
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

## Image pipeline

After adding or importing photos, run these scripts in order. Each is idempotent and safe to re-run.

| # | Script | Purpose |
|---|---|---|
| 1 | `scripts/optimize_images.sh` | Resize JPEGs to ≤1600px wide, quality 80 |
| 2 | `scripts/convert_webp.sh` | Make a full-size WebP sibling; deletes it if larger than the JPEG |
| 3 | `scripts/make_variants.sh` | Make the 200 / 480 / 800px WebP tiers |
| 4 | `scripts/add_img_dimensions.py` | Backfill `width`/`height` on `<img>` (prevents layout shift) |
| 5 | `scripts/responsive_images.py` | Rewrite markup: multi-tier `srcset` + `sizes` + LCP attributes on the first tile |
| 6 | `scripts/generate_thumbs.py` | Build list-page thumbnails and write `thumb:` into front matter |

The browser picks a tier from `srcset` using each cell's real width. `gallery.js` rewrites `<source sizes>` on every layout and zoom step, so zooming really does change resolution, not just layout. The lightbox always loads the full-size file. Average tier file sizes across all 443 photos: 200px → 6.7 KB, 480px → 32 KB, 800px → 78 KB.

Whole-post image payload. "Before" is what the page actually cost pre-branch — the full-size WebP for each photo, or the JPEG where no WebP came out smaller:

| Post | Photos | Before | all-200w | all-480w | all-800w |
|---|---:|---:|---:|---:|---:|
| Japan | 41 | 11.85 MB | 0.22 MB | 1.18 MB | 3.20 MB |
| Hangzhou | 14 | 11.00 MB | 0.16 MB | 1.06 MB | 2.78 MB |
| Switzerland | 79 | 17.36 MB | 0.47 MB | 2.25 MB | 5.41 MB |
| Germany | 296 | 67.76 MB | 1.96 MB | 9.11 MB | 21.46 MB |

Selection scales with device pixel ratio, because `sizes` is a CSS width. The 720px grid gives 233px cells at 3-across and 81px at 8-across, and roughly a fifth of tiles get a double-width slot (477px at 3-across) whenever there are at least 3 columns — the rule is pseudo-random, so short walls are lumpier (5 of 14 in Hangzhou, 10 of 41 in Japan, 59 of 296 in Germany). A normal cell therefore takes 480w at 1–2× and 800w at 3×; a double-width cell at 2× needs 954 device px, which skips 800w and lands on the full-size tier; 8-across fits in 200w at 1–2×. A phone defaults to 2-across (≈171px cells → 480w at 2×, 800w at 3×). The lightbox loads full size regardless of what the grid settled on.

Measured on the Japan post, whose 10 double-width tiles are what pull the 2× numbers up:

| Scenario | Payload | vs. before |
|---|---:|---:|
| Before this branch | 11.85 MB | — |
| 3-across, 1× display | 1.18 MB | 10× |
| 3-across, 2× display | 3.17 MB | 3.7× |
| 8-across, 2× display | 0.48 MB | 25× |
| 1-across, 2× display | 9.02 MB | 1.3× |

## Local development

```bash
gem install jekyll bundler
jekyll serve          # http://localhost:4000
```

Pushing to `master` triggers the GitHub Pages build automatically.

## Credits & license

Originally scaffolded from [Jekyll Now](https://github.com/barryclark/jekyll-now) by Barry Clark (MIT), since heavily customized. Released under the [MIT License](LICENSE).
