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

    var builtGrids = [];
    grids.forEach(function (group) {
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
      var cols = getComputedStyle(grid).gridTemplateColumns.split(' ').length;
      var colW = (grid.clientWidth - GAP * (cols - 1)) / cols;
      if (!colW || colW < 1) return;

      cells.forEach(function (cell, i) {
        // one in five gets a double-width slot (only when there is room)
        var wide = cols >= 3 && jitter(i) > 0.8;
        cell.classList.toggle('is-wide', wide);
        var slotW = wide ? colW * 2 + GAP : colW;
        var h = slotW * ratioOf(cell.querySelector('img'));
        cell.style.gridRowEnd = 'span ' + Math.max(2, Math.round((h + GAP) / (ROW + GAP)));

        // tilt: -1.6deg .. +1.6deg, plus a hair of vertical drift
        var tilt = (jitter(i + 100) - 0.5) * 3.2;
        var drift = (jitter(i + 200) - 0.5) * 5;
        cell.style.setProperty('--tilt', tilt.toFixed(2) + 'deg');
        cell.style.setProperty('--drift', drift.toFixed(1) + 'px');
      });
    }

    function layoutAll() { builtGrids.forEach(layout); }
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
