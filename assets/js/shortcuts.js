// Reading progress bar + keyboard shortcuts.
(function () {
  'use strict';

  var root = document.documentElement;

  // ------------------------------------------------ reading progress bar --
  function initProgress() {
    var entry = document.querySelector('article.post > .entry');
    if (!entry) return;   // only on single posts

    var bar = document.createElement('div');
    bar.className = 'read-progress';
    bar.innerHTML = '<span></span>';
    document.body.appendChild(bar);
    var fill = bar.firstElementChild;

    var ticking = false;
    function update() {
      var rect = entry.getBoundingClientRect();
      var total = rect.height - window.innerHeight;
      var done = -rect.top;
      var pct = total > 0 ? Math.min(1, Math.max(0, done / total)) : (rect.top <= 0 ? 1 : 0);
      fill.style.transform = 'scaleX(' + pct + ')';
      ticking = false;
    }
    function onScroll() {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(update);
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    update();
  }

  // ------------------------------------------------------------ shortcuts --
  var BASE = (document.querySelector('link[rel="alternate"]') || {}).href || '';
  function siteRoot() {
    // derive the site root from the RSS link (handles baseurl deployments)
    return BASE ? BASE.replace(/feed\.xml.*$/, '') : '/';
  }

  function go(path) { window.location.href = siteRoot() + path; }

  function isTyping(e) {
    var t = e.target;
    return t.isContentEditable ||
      /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName);
  }

  var HELP = [
    ['/', 'Search'],
    ['g h', 'Go to home (profile)'],
    ['g b', 'Go to blog'],
    ['g t', 'Go to tags'],
    ['g s', 'Go to search'],
    ['j / k', 'Next / previous heading'],
    ['t', 'Toggle light / dark theme'],
    ['c', 'Collapse / expand the table of contents'],
    ['?', 'Show this help'],
  ];

  function helpOverlay() {
    var el = document.querySelector('.shortcut-help');
    if (el) return el;
    el = document.createElement('div');
    el.className = 'shortcut-help';
    var rows = HELP.map(function (r) {
      return '<tr><td><kbd>' + r[0].split(' ').join('</kbd> <kbd>') +
             '</kbd></td><td>' + r[1] + '</td></tr>';
    }).join('');
    el.innerHTML = '<div class="shortcut-help-panel">' +
      '<h3>Keyboard shortcuts</h3><table>' + rows + '</table>' +
      '<p class="shortcut-help-hint">Press <kbd>Esc</kbd> to close</p></div>';
    el.addEventListener('click', function () { el.classList.remove('open'); });
    document.body.appendChild(el);
    return el;
  }

  function headings() {
    var entry = document.querySelector('article.post > .entry');
    return entry ? Array.prototype.slice.call(entry.querySelectorAll('h2, h3, h4')) : [];
  }

  function jumpHeading(dir) {
    var hs = headings();
    if (!hs.length) return;
    var offset = 100;
    var target = null;
    if (dir > 0) {
      for (var i = 0; i < hs.length; i++) {
        if (hs[i].getBoundingClientRect().top > offset + 5) { target = hs[i]; break; }
      }
    } else {
      for (var j = hs.length - 1; j >= 0; j--) {
        if (hs[j].getBoundingClientRect().top < offset - 5) { target = hs[j]; break; }
      }
    }
    if (target) window.scrollTo({ top: window.scrollY + target.getBoundingClientRect().top - offset, behavior: 'smooth' });
  }

  function initShortcuts() {
    var pending = null, timer = null;

    document.addEventListener('keydown', function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      var help = document.querySelector('.shortcut-help');
      if (e.key === 'Escape' && help && help.classList.contains('open')) {
        help.classList.remove('open');
        return;
      }

      // '/' focuses the search box when already on the search page
      if (isTyping(e)) return;

      // second key of a "g …" sequence
      if (pending === 'g') {
        pending = null;
        clearTimeout(timer);
        if (e.key === 'h') { go(''); return; }
        if (e.key === 'b') { go('blogs/'); return; }
        if (e.key === 't') { go('tags/'); return; }
        if (e.key === 's') { go('search/'); return; }
        return;
      }

      switch (e.key) {
        case 'g':
          pending = 'g';
          timer = setTimeout(function () { pending = null; }, 1500);
          break;
        case '/':
        case 's': {
          var box = document.getElementById('search-input');
          if (box) { e.preventDefault(); box.focus(); }
          else go('search/');
          break;
        }
        case 'j': jumpHeading(1); break;
        case 'k': jumpHeading(-1); break;
        case 't': {
          var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
          root.setAttribute('data-theme', next);
          try { localStorage.setItem('theme', next); } catch (err) {}
          break;
        }
        case 'c': {
          var toc = document.querySelector('.toc .toc-title');
          if (toc) toc.click();
          break;
        }
        case '?': helpOverlay().classList.add('open'); break;
      }
    });
  }

  window.addEventListener('DOMContentLoaded', function () {
    initProgress();
    initShortcuts();
  });
})();
