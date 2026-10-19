import sys, os, re, json, subprocess, tempfile
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TPL = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<style>
@page { size: A4; margin: 18mm 18mm 20mm 18mm; }
body { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 10.5pt; line-height: 1.75; color:#111; }
h1 { font-size: 17pt; border-bottom: 2px solid #333; padding-bottom: 6px; margin-top:0; }
h2 { font-size: 13pt; margin-top: 20px; border-left: 4px solid #2b6cb0; padding-left: 8px; break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 14px; break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 4px 7px; vertical-align: top; }
th { background: #f0f3f7; }
blockquote { margin: 8px 0; padding: 6px 12px; background: #f7f7f2; border-left: 3px solid #999; }
code { font-size: 9pt; background:#f3f3f3; padding:0 3px; }
a { color: #2b6cb0; word-break: break-all; }
.katex-display { margin: 10px 0; overflow: hidden; }
hr { border: 0; border-top: 1px solid #ccc; }
li { margin: 2px 0; }
.foot { margin-top: 24px; font-size: 8.5pt; color: #777; border-top:1px solid #ddd; padding-top:6px; }
.memo { margin-top: 18px; break-inside: avoid; }
.memo div { border-bottom: 1px dashed #bbb; height: 26px; }
</style></head><body><div id="c"></div>
<div class="memo"><h2>✍️ 나의 메모</h2><div></div><div></div><div></div><div></div><div></div><div></div></div>
<div class="foot">AI Life 논문 서재 · 원리와 수학으로 읽는 인공지능</div>
<script>
const src = __SRC__;
const maths = [];
let s = src.replace(/\$\$([\s\S]+?)\$\$/g, (m, t) => { maths.push([t, true]); return "@@M" + (maths.length-1) + "@@"; });
s = s.replace(/\$([^\$\n]+?)\$/g, (m, t) => { maths.push([t, false]); return "@@M" + (maths.length-1) + "@@"; });
let html = marked.parse(s);
html = html.replace(/@@M(\d+)@@/g, (m, i) => katex.renderToString(maths[+i][0], {displayMode: maths[+i][1], throwOnError: false}));
document.getElementById("c").innerHTML = html;
</script></body></html>"""
plain = "--plain" in sys.argv  # 해설지용: 메모란·꼬리말 없이
for md in [a for a in sys.argv[1:] if a != "--plain"]:
    src = open(md, encoding="utf-8").read()
    html = TPL.replace("__SRC__", json.dumps(src))
    if plain:
        html = re.sub(r'<div class="memo">.*?</div>\n<div class="foot">.*?</div>', "", html, flags=re.S)
    tmp = os.path.join(tempfile.gettempdir(), "ai_life_render.html")
    open(tmp, "w", encoding="utf-8").write(html)
    out = md[:-3] + ".pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=15000", f"--print-to-pdf={out}", tmp],
                   capture_output=True, timeout=120)
    print(("OK " if os.path.exists(out) else "FAIL ") + out)
