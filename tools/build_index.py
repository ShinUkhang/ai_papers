"""논문 서재 index.html 생성기.

사용법: python3 tools/build_index.py
각 분야 폴더(01_~05_)의 리뷰 .md를 읽어, 주차별·분야별 카드 목록 index.html을 만든다.
"""
import glob
import html
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATEGORIES = {
    "01_인공지능_수학": "인공지능 수학",
    "02_기계학습": "기계학습",
    "03_딥러닝": "딥러닝",
    "04_컴퓨터_교육": "컴퓨터 교육",
    "05_네트워크": "네트워크",
}
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def parse(path):
    text = open(path, encoding="utf-8").read()
    lines = text.split("\n")
    info = {"title": lines[0].lstrip("# ").strip()}
    m = re.search(r"원제: \*(.+?)\*", text)
    info["original"] = m.group(1) if m else ""
    for line in lines:
        if line.startswith("| ") and line.count("|") >= 3:
            key, val = [c.strip() for c in line.strip("|").split("|", 1)]
            info.setdefault(key, val.strip(" |"))
    m = re.search(r"## 한 줄 요약\n(.+)", text)
    info["summary"] = m.group(1).replace("**", "") if m else ""
    return info


def md_links(s):
    s = s.replace("*", "")
    out, last = [], 0
    for m in LINK.finditer(s):
        out.append(html.escape(s[last:m.start()]))
        out.append(f'<a href="{html.escape(m.group(2))}" target="_blank" rel="noopener">{html.escape(m.group(1))}</a>')
        last = m.end()
    out.append(html.escape(s[last:]))
    return "".join(out)


def card(folder, path):
    info = parse(path)
    base = os.path.basename(path)[:-3]
    review_pdf = f"{folder}/{base}.pdf"
    venue = info.get("학회") or info.get("학술지") or ""
    return f"""<article class="card" data-cat="{CATEGORIES[folder]}">
  <div class="tag">{CATEGORIES[folder]} · {html.escape(info.get('난이도', ''))}</div>
  <h3>{html.escape(info['title'])}</h3>
  <p class="orig">{html.escape(info['original'])}</p>
  <p class="meta">{html.escape(info.get('저자', ''))}<br>{md_links(venue)}</p>
  <p class="sum">{html.escape(info['summary'])}</p>
  <p class="unit">🔗 ai_life 연결: {html.escape(info.get('ai_life 연결 단원', '-'))}</p>
  <div class="btns"><a class="btn primary" href="{html.escape(review_pdf)}" target="_blank">한국어 리뷰지 PDF</a>
  <span class="btn">{md_links(info.get('PDF 원문', ''))}</span></div>
</article>"""


def main():
    weeks = {}
    for folder in CATEGORIES:
        for path in sorted(glob.glob(os.path.join(ROOT, folder, "*.md"))):
            week = os.path.basename(path).split("_")[0]
            weeks.setdefault(week, []).append(card(folder, path))
    sections = []
    for week in sorted(weeks, reverse=True):
        guide = f"00_주간_읽기목록/{week}_이번주_읽기목록.pdf"
        sections.append(f"""<section class="week"><h2>{week[:4]}년 {int(week[6:])}주차
  <a class="guide" href="{guide}" target="_blank">📅 이번 주 읽기 가이드 PDF</a></h2>
  <div class="grid">{''.join(weeks[week])}</div></section>""")
    filters = "".join(f'<button data-f="{c}">{c}</button>' for c in CATEGORIES.values())
    page = TEMPLATE.replace("__FILTERS__", filters).replace("__SECTIONS__", "\n".join(sections))
    open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write(page)
    print("index.html:", sum(len(v) for v in weeks.values()), "papers,", len(weeks), "weeks")


TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Life 논문 서재</title>
<style>
:root { --bg:#f7f8fb; --card:#fff; --ink:#1a202c; --sub:#5a6475; --line:#e2e6ee; --accent:#2b6cb0; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#12151b; --card:#1b2029; --ink:#e7ebf2; --sub:#9aa5b8; --line:#2c3341; --accent:#7db3f0; } }
:root[data-theme="dark"] { --bg:#12151b; --card:#1b2029; --ink:#e7ebf2; --sub:#9aa5b8; --line:#2c3341; --accent:#7db3f0; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:"Apple SD Gothic Neo","Noto Sans KR",sans-serif; line-height:1.6; }
header { padding:40px 16px 20px; max-width:1100px; margin:auto; }
header h1 { margin:0 0 6px; font-size:28px; }
header p { margin:0; color:var(--sub); }
nav { display:flex; gap:12px; flex-wrap:wrap; margin-top:14px; font-size:14px; }
nav a { color:var(--accent); }
.filters { max-width:1100px; margin:0 auto; padding:0 16px; display:flex; gap:8px; flex-wrap:wrap; }
.filters button { border:1px solid var(--line); background:var(--card); color:var(--ink); border-radius:20px; padding:6px 14px; cursor:pointer; font:inherit; font-size:14px; }
.filters button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
main { max-width:1100px; margin:auto; padding:0 16px 60px; }
.week h2 { font-size:20px; margin:32px 0 12px; display:flex; flex-wrap:wrap; gap:12px; align-items:center; }
.guide { font-size:14px; font-weight:normal; color:var(--accent); }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; display:flex; flex-direction:column; }
.card h3 { margin:6px 0 2px; font-size:17px; }
.tag { font-size:12px; color:var(--accent); font-weight:600; }
.orig { margin:0; font-style:italic; color:var(--sub); font-size:13px; }
.meta { font-size:12.5px; color:var(--sub); margin:8px 0; }
.sum { font-size:14px; margin:4px 0 8px; flex:1; }
.unit { font-size:12.5px; color:var(--sub); margin:0 0 10px; }
.btns { display:flex; flex-wrap:wrap; gap:8px; font-size:13px; }
.btn { border:1px solid var(--line); border-radius:8px; padding:5px 10px; }
.btn.primary { background:var(--accent); color:#fff; border-color:var(--accent); text-decoration:none; }
a { color:var(--accent); }
</style></head><body>
<header>
  <h1>📚 AI Life 논문 서재</h1>
  <p>원리와 수학으로 읽는 인공지능 — 매주 10편, 한국어 리뷰지와 원문 PDF</p>
  <nav><a href="00_참고도서/참고도서_추천.pdf" target="_blank">참고 도서 추천</a>
  <a href="https://shinukhang.github.io/ai_life/">← AI Life 커리큘럼으로</a></nav>
</header>
<div class="filters"><button class="on" data-f="all">전체</button>__FILTERS__</div>
<main>
__SECTIONS__
</main>
<script>
document.querySelectorAll('.filters button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.filters button').forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('.card').forEach(c => c.style.display = (b.dataset.f === 'all' || c.dataset.cat === b.dataset.f) ? '' : 'none');
});
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
