// Post enhancements: code-block copy buttons + auto table of contents.
// Runs only on single post pages (skipped on the index).
(function () {
  'use strict';

  window.addEventListener('DOMContentLoaded', function () {

    // ---------- 1. copy button on every code block ----------
    document.querySelectorAll('div.highlight').forEach(function (block) {
      var pre = block.querySelector('pre');
      if (!pre) return;
      block.classList.add('has-copy');

      var btn = document.createElement('button');
      btn.className = 'code-copy';
      btn.type = 'button';
      btn.textContent = 'Copy';
      btn.addEventListener('click', function () {
        navigator.clipboard.writeText(pre.innerText).then(function () {
          btn.textContent = '✓ Copied';
          btn.classList.add('done');
          setTimeout(function () {
            btn.textContent = 'Copy';
            btn.classList.remove('done');
          }, 1600);
        });
      });
      block.appendChild(btn);
    });

    // ---------- 2. table of contents ----------
    if (document.querySelector('.posts')) return; // index page: skip
    var entry = document.querySelector('article.post > .entry');
    if (!entry) return;
    var heads = entry.querySelectorAll('h2, h3');
    if (heads.length < 3) return;

    // unique, CJK-friendly ids
    var seen = {};
    heads.forEach(function (h) {
      if (h.id) return;
      var slug = h.textContent.trim().toLowerCase()
        .replace(/[^\w\u4e00-\u9fff\u3040-\u30ff-]+/g, '-')
        .replace(/^-+|-+$/g, '') || 'section';
      if (seen[slug] !== undefined) slug += '-' + (++seen[slug]);
      else seen[slug] = 0;
      h.id = slug;
    });

    function buildList() {
      var ul = document.createElement('ul');
      heads.forEach(function (h) {
        var li = document.createElement('li');
        li.className = 'toc-' + h.tagName.toLowerCase();
        var a = document.createElement('a');
        a.href = '#' + h.id;
        a.textContent = h.textContent;
        li.appendChild(a);
        ul.appendChild(li);
      });
      return ul;
    }

    // desktop: floating sidebar
    var aside = document.createElement('nav');
    aside.className = 'toc';
    aside.innerHTML = '<div class="toc-title">目录</div>';
    aside.appendChild(buildList());
    document.body.appendChild(aside);

    // mobile: collapsible list above the content
    var details = document.createElement('details');
    details.className = 'toc-mobile';
    details.innerHTML = '<summary>目录</summary>';
    details.appendChild(buildList());
    entry.parentNode.insertBefore(details, entry);

    // highlight the section currently in view
    var links = aside.querySelectorAll('a');
    var byId = {};
    links.forEach(function (a) { byId[decodeURIComponent(a.hash.slice(1))] = a; });
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          links.forEach(function (a) { a.classList.remove('active'); });
          var a = byId[e.target.id];
          if (a) a.classList.add('active');
        }
      });
    }, { rootMargin: '-90px 0px -70% 0px' });
    heads.forEach(function (h) { observer.observe(h); });
  });
})();
