// Mermaid diagrams — loaded only on posts with `mermaid: true`.
//
// Rouge renders ```mermaid fences as <div class="highlight"><pre>…</pre></div>,
// so the raw source is lifted out of those blocks first, then Mermaid renders
// it. Re-renders on theme switch so diagrams match light/dark mode.
(function () {
  'use strict';

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'dark'
      ? 'dark' : 'default';
  }

  window.addEventListener('DOMContentLoaded', function () {
    // Collect mermaid sources and replace the code blocks with render targets.
    // Rouge has no mermaid lexer, so kramdown falls back to a plain
    // <pre><code class="language-mermaid"> block — handle that shape first,
    // and still accept div.highlight in case a lexer appears later.
    var sources = [];

    function claim(block, code) {
      var target = document.createElement('div');
      target.className = 'mermaid-figure';
      block.parentNode.replaceChild(target, block);
      sources.push({ el: target, code: code });
    }

    document.querySelectorAll('pre > code.language-mermaid').forEach(function (code) {
      claim(code.parentNode, code.textContent.trim());
    });

    document.querySelectorAll('div.highlight').forEach(function (block) {
      var pre = block.querySelector('pre');
      if (!pre) return;
      var code = (pre.innerText || pre.textContent || '').trim();
      if (!code) return;
      var isMermaid = block.className.indexOf('language-mermaid') >= 0 ||
        /^(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|journey|gantt|pie|gitGraph|mindmap|timeline|quadrantChart|xychart)/.test(code);
      if (!isMermaid) return;
      claim(block, code);
    });

    if (!sources.length) return;

    import('https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs')
      .then(function (mod) {
        var mermaid = mod.default;
        var seq = 0;

        function renderAll() {
          mermaid.initialize({
            startOnLoad: false,
            theme: currentTheme(),
            fontFamily: 'inherit',
          });
          sources.forEach(function (s) {
            var id = 'mmd-' + (seq++);
            mermaid.render(id, s.code)
              .then(function (res) { s.el.innerHTML = res.svg; })
              .catch(function (err) {
                s.el.innerHTML = '<pre class="mermaid-error">' +
                  String(err.message || err).replace(/[<>&]/g, '') + '</pre>';
              });
          });
        }

        renderAll();

        // redraw with matching colors when the reader flips the theme
        new MutationObserver(renderAll).observe(document.documentElement, {
          attributes: true, attributeFilter: ['data-theme'],
        });
      })
      .catch(function () {
        sources.forEach(function (s) {
          s.el.innerHTML = '<pre>' + s.code + '</pre>';  // graceful fallback
        });
      });
  });
})();
