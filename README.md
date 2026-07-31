# zty2004.github.io

Personal homepage & blog of **Tianyou Zuo (fztym)** — built with [Jekyll](https://jekyllrb.com/) on GitHub Pages.

🔗 **Live site:** <https://zty2004.github.io>

## Features

- 🌅 **Monet sunset theme** — hand-crafted palette inspired by *Impression, Sunrise*, with drifting light-ray background, canvas grain texture, and a cursive gradient signature
- 🌗 **Light / dark mode** — sun–moon slider fixed at the top-right, follows system preference, remembered via `localStorage`
- 🎓 **Academic homepage** — profile, news, research interests, education timeline at `/tianyouzuo/`
- 🏷️ **Tags** — per-post tags with a tag-cloud archive at `/tags/`
- 🧮 **LaTeX math** — MathJax 3 loaded on demand via `math: true` front matter (always write formulas as `$$...$$`)
- 💬 **Comments & stats** — [giscus](https://giscus.app) (GitHub Discussions) with theme sync, busuanzi visitor counters
- ⚡ **Performance** — photos batch-compressed (1.6 GB → 219 MB), lazy-loaded images, minified CSS

## Repository layout

```
├── _config.yml       # site-wide configuration
├── index.html        # post list (home page)
├── _pages/           # standalone pages: about, tags, 404
├── _posts/           # blog posts (YYYY-M-D-Title.md)
├── _layouts/         # default / page / post templates
├── _includes/        # meta, icons, analytics, giscus snippets
├── _sass/            # theme partials (variables, reset, highlights, icons)
├── assets/css/       # main stylesheet entry (style.scss)
├── images/           # post photos (compressed for web)
├── scripts/          # maintenance scripts (image optimizer)
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
