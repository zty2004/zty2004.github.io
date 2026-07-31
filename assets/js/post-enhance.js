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

    // ---------- 2. table of contents (collapsible tree) ----------
    if (document.querySelector('.posts')) return; // index page: skip
    var entry = document.querySelector('article.post > .entry');
    if (!entry) return;
    var heads = Array.prototype.slice.call(entry.querySelectorAll('h2, h3, h4'));
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

    // nest headings into a tree by level (h2 > h3 > h4)
    function buildTree() {
      var root = { level: 1, children: [] };
      var stack = [root];
      heads.forEach(function (h) {
        var node = { h: h, level: +h.tagName[1], children: [] };
        while (stack[stack.length - 1].level >= node.level) stack.pop();
        stack[stack.length - 1].children.push(node);
        stack.push(node);
      });
      return root;
    }

    // collapsible=true renders fold toggles; false renders a plain nested list
    function renderList(nodes, collapsible) {
      var ul = document.createElement('ul');
      nodes.forEach(function (n) {
        var li = document.createElement('li');
        li.className = 'toc-' + n.h.tagName.toLowerCase();
        var row = document.createElement('div');
        row.className = 'toc-row';

        if (collapsible) {
          if (n.children.length) {
            var tg = document.createElement('button');
            tg.type = 'button';
            tg.className = 'toc-toggle';
            tg.setAttribute('aria-label', 'toggle section');
            tg.textContent = '▸';
            tg.addEventListener('click', function (e) {
              e.stopPropagation();
              li.classList.toggle('open');
            });
            row.appendChild(tg);
          } else {
            var pad = document.createElement('span');
            pad.className = 'toc-toggle toc-pad';
            row.appendChild(pad);
          }
        }

        var a = document.createElement('a');
        a.href = '#' + n.h.id;
        a.textContent = n.h.textContent;
        row.appendChild(a);
        li.appendChild(row);
        if (n.children.length) li.appendChild(renderList(n.children, collapsible));
        ul.appendChild(li);
      });
      return ul;
    }

    var tree = buildTree();

    // desktop: right-hand sidebar, top level visible, deeper levels fold out;
    // the whole panel can also be minimized (state remembered)
    var aside = document.createElement('nav');
    aside.className = 'toc';
    var titleBar = document.createElement('button');
    titleBar.type = 'button';
    titleBar.className = 'toc-title';
    titleBar.title = '折叠/展开目录';
    titleBar.innerHTML = '<span class="toc-title-text">目录</span>' +
                         '<span class="toc-fold"></span>';
    aside.appendChild(titleBar);
    aside.appendChild(renderList(tree.children, true));
    document.body.appendChild(aside);

    var fold = titleBar.querySelector('.toc-fold');
    function setFolded(folded) {
      aside.classList.toggle('collapsed', folded);
      fold.textContent = folded ? '☰' : '−';
      titleBar.setAttribute('aria-expanded', folded ? 'false' : 'true');
      try { localStorage.setItem('tocCollapsed', folded ? '1' : ''); } catch (e) {}
    }
    var saved = false;
    try { saved = localStorage.getItem('tocCollapsed') === '1'; } catch (e) {}
    setFolded(saved);
    titleBar.addEventListener('click', function () {
      setFolded(!aside.classList.contains('collapsed'));
    });

    // mobile: collapsible list above the content (same foldable tree)
    var details = document.createElement('details');
    details.className = 'toc-mobile';
    details.innerHTML = '<summary>目录</summary>';
    details.appendChild(renderList(tree.children, true));
    entry.parentNode.insertBefore(details, entry);

    // tapping a link closes the mobile panel so it stops covering the text
    details.addEventListener('click', function (e) {
      if (e.target.tagName === 'A') details.open = false;
    });

    // highlight the section in view and unfold its ancestors (both TOCs)
    var links = [].concat(
      Array.prototype.slice.call(aside.querySelectorAll('a')),
      Array.prototype.slice.call(details.querySelectorAll('a'))
    );
    var byId = {};
    links.forEach(function (a) {
      var id = decodeURIComponent(a.hash.slice(1));
      (byId[id] = byId[id] || []).push(a);
    });
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        var hits = byId[e.target.id];
        if (!hits) return;
        links.forEach(function (x) { x.classList.remove('active'); });
        hits.forEach(function (a) {
          a.classList.add('active');
          var li = a.closest('li');
          while (li) {
            li.classList.add('open');
            li = li.parentElement ? li.parentElement.closest('li') : null;
          }
        });
      });
    }, { rootMargin: '-90px 0px -70% 0px' });
    heads.forEach(function (h) { observer.observe(h); });
  });
})();
