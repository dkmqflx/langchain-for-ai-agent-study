/* 코드 블록의 '주석'만 찾아 다른 색(.tok-c)으로 칠한다.
   - 대상: <pre><code> 안의 # 주석과 파이썬 """docstring"""
   - 제외: <code class="nohl"> (예: 마크다운 템플릿 — 거기서 #은 주석이 아니라 제목이다)
   문법 하이라이팅 전체가 아니라 주석만 다룬다. 읽는 데 필요한 건 그 구분뿐이라서. */
(function () {
  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // 한 줄을 [코드, 주석]으로 가른다. 따옴표 안의 #은 주석이 아니므로 건너뛴다.
  function splitComment(line) {
    var quote = null;
    for (var i = 0; i < line.length; i++) {
      var ch = line[i];
      if (quote) {
        if (ch === "\\") { i++; continue; }   // 이스케이프된 문자는 통째로 건너뛴다
        if (ch === quote) quote = null;
      } else if (ch === '"' || ch === "'") {
        quote = ch;
      } else if (ch === "#") {
        return [line.slice(0, i), line.slice(i)];
      }
    }
    return [line, ""];
  }

  function wrap(s) {
    return '<span class="tok-c">' + esc(s) + "</span>";
  }

  document.querySelectorAll("pre > code").forEach(function (code) {
    if (code.classList.contains("nohl")) return;

    var inDocstring = false;
    var html = code.textContent.split("\n").map(function (line) {
      var triples = (line.match(/"""/g) || []).length;

      if (inDocstring) {
        if (triples % 2 === 1) inDocstring = false;   // 홀수 개면 여기서 닫힌다
        return wrap(line);
      }
      if (triples > 0) {
        if (triples % 2 === 1) inDocstring = true;    // 한 줄짜리 docstring이면 열리지 않는다
        return wrap(line);
      }

      var parts = splitComment(line);
      return esc(parts[0]) + (parts[1] ? wrap(parts[1]) : "");
    });

    code.innerHTML = html.join("\n");
  });
})();
