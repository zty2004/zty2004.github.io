// Gallery mode: turn runs of consecutive images into a grid, with a lightweight
// lightbox that supports arrow-key navigation. Activated by `gallery: true` in
// a post's front matter (the layout sets data-gallery on the article).
(function () {
  'use strict';

  window.addEventListener('DOMContentLoaded', function () {
    var article = document.querySelector('article.post[data-gallery="true"]');
    if (!article) return;
    var entry = article.querySelector('.entry');
    if (!entry) return;

    // ---------- 1. group consecutive images into grids ----------
    // kramdown emits raw <picture>/<img> block tags as direct children of
    // .entry (no <p> wrapper), while Markdown-syntax images end up inside a
    // paragraph. Accept both shapes.
    function imageOf(node) {
      if (node.nodeType !== 1) return null;
      if (node.tagName === 'PICTURE') return node.querySelector('img');
      if (node.tagName === 'IMG') return node;
      if (node.tagName === 'P' && node.textContent.trim() === '') {
        var imgs = node.querySelectorAll('img');
        if (imgs.length === 1) return imgs[0];
      }
      return null;
    }

    var children = Array.prototype.slice.call(entry.children);
    var run = [];
    var grids = [];

    function flush() {
      if (run.length >= 2) grids.push(run.slice());
      run = [];
    }
    children.forEach(function (node) {
      if (imageOf(node)) run.push(node);
      else flush();
    });
    flush();

    // Every image in the post belongs to the lightbox, in document order —
    // grids are only a layout choice, so standalone photos between paragraphs
    // must stay navigable with the arrow keys too.
    var items = Array.prototype.slice.call(entry.querySelectorAll('img'));

    // Photo-only posts get the loose, hand-laid wall. Posts that mix prose with
    // photos keep the plain vertical flow so pictures stay next to the text
    // that discusses them — only the lightbox is added there.
    var proseLength = 0;
    children.forEach(function (node) {
      if (!imageOf(node)) proseLength += node.textContent.trim().length;
    });
    var photoOnly = proseLength < 40 && items.length >= 4;

    var builtGrids = [];
    if (photoOnly) grids.forEach(function (group) {
      var grid = document.createElement('div');
      grid.className = 'gallery';
      entry.insertBefore(grid, group[0]);
      group.forEach(function (node) {
        // move the media itself: the node may BE the <picture>, or wrap one
        var media = node.tagName === 'P'
          ? (node.querySelector('picture') || node.querySelector('img'))
          : node;
        var cell = document.createElement('button');
        cell.type = 'button';
        cell.className = 'gallery-item';
        cell.appendChild(media);
        grid.appendChild(cell);
        if (node.tagName === 'P') node.remove();
      });
      builtGrids.push(grid);
    });

    // ---------- Mission-Control-ish mosaic ----------
    // Photos keep their real aspect ratio and flow into a masonry grid; a few
    // get a wider slot and every tile is nudged a fraction of a degree so the
    // wall feels hand-laid rather than machine-stamped.
    var ROW = 6;   // grid-auto-rows, px  (keep in sync with the stylesheet)
    var GAP = 10;  // gallery gap, px

    // Zoom, Photos.app style: fewer columns = bigger photos. Pinch on a
    // trackpad/touchscreen, Cmd/Ctrl + wheel, or the +/− buttons.
    var MIN_COLS = 1, MAX_COLS = 8;
    var narrow = !!(window.matchMedia && window.matchMedia('(max-width: 900px)').matches);
    var cols = narrow ? 2 : 3;
    try {
      var saved = parseInt(localStorage.getItem('galleryCols'), 10);
      if (saved >= MIN_COLS && saved <= MAX_COLS) cols = saved;
    } catch (e) {}

    function ratioOf(img) {
      var w = parseFloat(img.getAttribute('width')) || img.naturalWidth || 4;
      var h = parseFloat(img.getAttribute('height')) || img.naturalHeight || 3;
      return h / w;
    }

    // deterministic pseudo-random so a reload doesn't reshuffle the wall
    function jitter(i) {
      var x = Math.sin((i + 1) * 12.9898) * 43758.5453;
      return x - Math.floor(x);           // 0..1
    }

    function layout(grid) {
      var cells = Array.prototype.slice.call(grid.children);
      var colW = (grid.clientWidth - GAP * (cols - 1)) / cols;
      if (!colW || colW < 1) return;

      cells.forEach(function (cell, i) {
        // one in five gets a double-width slot (only when there is room)
        var wide = cols >= 3 && jitter(i) > 0.8;
        cell.classList.toggle('is-wide', wide);
        var slotW = wide ? colW * 2 + GAP : colW;
        var h = slotW * ratioOf(cell.querySelector('img'));
        cell.style.gridRowEnd = 'span ' + Math.max(2, Math.round((h + GAP) / (ROW + GAP)));

        // tilt: -1.6deg .. +1.6deg, plus a hair of vertical drift.
        // A single-column view is "inspect one photo" mode — keep those straight.
        var tilt = cols === 1 ? 0 : (jitter(i + 100) - 0.5) * 3.2;
        var drift = cols === 1 ? 0 : (jitter(i + 200) - 0.5) * 5;
        cell.style.setProperty('--tilt', tilt.toFixed(2) + 'deg');
        cell.style.setProperty('--drift', drift.toFixed(1) + 'px');
      });
    }

    function layoutAll() { builtGrids.forEach(layout); }

    function setCols(n, zoomEl) {
      n = Math.min(MAX_COLS, Math.max(MIN_COLS, n));
      if (n === cols) return;
      cols = n;
      try { localStorage.setItem('galleryCols', String(cols)); } catch (e) {}
      builtGrids.forEach(function (g) { g.style.setProperty('--cols', cols); });
      layoutAll();
      if (zoomEl) zoomEl.textContent = cols;
    }

    builtGrids.forEach(function (grid) {
      grid.style.setProperty('--cols', cols);

      // --- zoom controls (discoverability for the gesture) ---
      var bar = document.createElement('div');
      bar.className = 'gallery-zoom';
      bar.innerHTML =
        '<button type="button" class="gz-out" aria-label="Show more photos">−</button>' +
        '<span class="gz-value">' + cols + '</span>' +
        '<button type="button" class="gz-in" aria-label="Show bigger photos">+</button>' +
        '<span class="gz-hint">双指捏合或 ⌘/Ctrl + 滚轮缩放</span>';
      grid.parentNode.insertBefore(bar, grid);
      var valueEl = bar.querySelector('.gz-value');

      bar.querySelector('.gz-in').addEventListener('click', function () { setCols(cols - 1, valueEl); });
      bar.querySelector('.gz-out').addEventListener('click', function () { setCols(cols + 1, valueEl); });

      // --- trackpad pinch / Cmd+wheel: the browser reports ctrlKey ---
      grid.addEventListener('wheel', function (e) {
        if (!e.ctrlKey && !e.metaKey) return;   // plain scrolling must still scroll
        e.preventDefault();
        setCols(cols + (e.deltaY > 0 ? 1 : -1), valueEl);
      }, { passive: false });

      // --- touchscreen pinch ---
      var startDist = 0, startCols = cols;
      function spread(t) {
        var dx = t[0].clientX - t[1].clientX, dy = t[0].clientY - t[1].clientY;
        return Math.sqrt(dx * dx + dy * dy);
      }
      grid.addEventListener('touchstart', function (e) {
        if (e.touches.length !== 2) return;
        startDist = spread(e.touches);
        startCols = cols;
      }, { passive: true });
      grid.addEventListener('touchmove', function (e) {
        if (e.touches.length !== 2 || !startDist) return;
        e.preventDefault();
        var scale = spread(e.touches) / startDist;
        setCols(Math.round(startCols / scale), valueEl);
      }, { passive: false });
      grid.addEventListener('touchend', function () { startDist = 0; }, { passive: true });
    });

    layoutAll();

    // re-flow on resize (column count and width both change)
    var rt = null;
    window.addEventListener('resize', function () {
      clearTimeout(rt);
      rt = setTimeout(layoutAll, 150);
    });
    // images may report different intrinsic sizes than the attributes claim
    window.addEventListener('load', layoutAll);

    if (!items.length) return;

    // clicking any image — gridded or standalone — opens the viewer
    entry.addEventListener('click', function (e) {
      var img = e.target.closest('img');
      if (!img || img.closest('a')) return;
      var i = items.indexOf(img);
      if (i >= 0) open(i);
    });

    // ---------- 2. minimal lightbox with prev/next ----------
    var box = document.createElement('div');
    box.className = 'lightbox';
    box.innerHTML =
      '<button class="lightbox-close" aria-label="Close">✕</button>' +
      '<button class="lightbox-prev" aria-label="Previous">‹</button>' +
      '<img class="lightbox-img" alt="">' +
      '<button class="lightbox-next" aria-label="Next">›</button>' +
      '<div class="lightbox-count"></div>';
    document.body.appendChild(box);

    var bigImg = box.querySelector('.lightbox-img');
    var counter = box.querySelector('.lightbox-count');
    var current = -1;

    function show(i) {
      current = (i + items.length) % items.length;
      var src = items[current].currentSrc || items[current].src;
      bigImg.src = src;
      bigImg.alt = items[current].alt || '';
      counter.textContent = (current + 1) + ' / ' + items.length;
    }
    function open(i) {
      if (i < 0) return;
      show(i);
      box.classList.add('open');
      document.body.style.overflow = 'hidden';
    }
    function close() {
      box.classList.remove('open');
      document.body.style.overflow = '';
      bigImg.src = '';
    }

    box.querySelector('.lightbox-close').addEventListener('click', close);
    box.querySelector('.lightbox-prev').addEventListener('click', function (e) {
      e.stopPropagation(); show(current - 1);
    });
    box.querySelector('.lightbox-next').addEventListener('click', function (e) {
      e.stopPropagation(); show(current + 1);
    });
    box.addEventListener('click', function (e) {
      if (e.target === box || e.target === bigImg) close();
    });

    document.addEventListener('keydown', function (e) {
      if (!box.classList.contains('open')) return;
      if (e.key === 'Escape') close();
      else if (e.key === 'ArrowLeft') show(current - 1);
      else if (e.key === 'ArrowRight') show(current + 1);
    });

    // swipe on touch devices
    var startX = null;
    box.addEventListener('touchstart', function (e) { startX = e.touches[0].clientX; }, { passive: true });
    box.addEventListener('touchend', function (e) {
      if (startX === null) return;
      var dx = e.changedTouches[0].clientX - startX;
      if (Math.abs(dx) > 50) show(current + (dx < 0 ? 1 : -1));
      startX = null;
    });
  });
})();
