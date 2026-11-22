"""Local test of the weekly page's scripts: a stand-in for the Data type's `dash` API (data, draw, colors) fed with the same CSV
files, d3 v7 from cdnjs, and the page's HTML; rendered with headless Chrome at desktop and phone width, light and dark. Script
errors are printed into the page in red so the screenshots show them.

  /home/jli199/boptim-agent/.venv/bin/python reports/weekly_1121/harness.py
"""
import csv
import glob
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
data = {}
for p in sorted(glob.glob(os.path.join(HERE, "data", "*.csv"))):
    with open(p) as f:
        rows = list(csv.DictReader(f))
    data[os.path.basename(p)[:-4]] = [{k: (v if v != "" else None) for k, v in r.items()} for r in rows]
page = open(os.path.join(HERE, "index.html")).read().replace('<meta name="dash-one-writer">', "")
TOKENS_LIGHT = """--color-fg:#1b1b1a; --color-fg-muted:#5f5e59; --color-bg:#fafaf8; --color-panel:#ffffff; --color-border-line:#e3e2dd;
  --font-anthropic-sans: Inter, Arial, sans-serif; --font-anthropic-serif: Georgia, serif;
  --cds-font-size-caption:12px; --cds-font-size-body:15px; --cds-font-size-heading:20px; --cds-font-size-title:30px; --cds-font-weight-medium:600;
  --cds-gap-xs:4px; --cds-gap-sm:8px; --cds-gap-md:12px; --cds-gap-lg:20px; --cds-gap-xl:32px; --cds-pad-md:12px; --cds-pad-lg:16px; --cds-radius:8px;
  --cds-chart-muted:#a3a29c; --cds-chart-grid:#ecebe7; --cds-chart-axis:#c9c8c2; --cds-chart-reference:#8f8e88; --cds-text-secondary:#6b6a65;
  --cds-surface-popover:#ffffff; --cds-border:#dddcd6; --cds-shadow-popover:0 2px 8px rgba(0,0,0,.12);
  --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a;"""
TOKENS_DARK = """--color-fg:#eeede9; --color-fg-muted:#a9a8a2; --color-bg:#1a1a19; --color-panel:#232321; --color-border-line:#3a3a37;
  --cds-chart-muted:#77766f; --cds-chart-grid:#2f2f2c; --cds-chart-axis:#4a4a46; --cds-chart-reference:#8f8e88; --cds-text-secondary:#a9a8a2;
  --cds-surface-popover:#2b2b29; --cds-border:#444440; --c1:#3987e5; --c2:#d95926; --c3:#199e70;"""


def build(dark):
    mock = f"""<script>
window.__errs = [];
window.addEventListener('error', function (e) {{ document.body.insertAdjacentHTML('beforeend', '<pre style="color:red">' + e.message + '</pre>'); }});
window.dash = (function () {{
  const DATA = {json.dumps(data)};
  const draws = {{}};
  return {{
    colors: ['var(--c1)', 'var(--c2)', 'var(--c3)'],
    data: function (id) {{ return DATA[id] ? {{status: 'ok', data: DATA[id]}} : {{status: 'missing', data: []}}; }},
    draw: function (name, fn) {{ draws[name] = fn; }},
    calc: function (id, spec) {{ DATA[id] = spec.fn.apply(null, spec.inputs.map(function (i) {{ return DATA[i] || []; }})); }},
    onData: function (fn) {{ fn(); }},
    _run: function () {{
      Object.keys(draws).forEach(function (name) {{
        document.querySelectorAll('[data-draw="' + name + '"]').forEach(function (el) {{
          try {{ draws[name](el); }} catch (e) {{ document.body.insertAdjacentHTML('beforeend', '<pre style="color:red">draw ' + name + ': ' + e + '</pre>'); }}
        }});
      }});
    }}
  }};
}})();
</script>"""
    head = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            f'<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>'
            f'<style>:root{{{TOKENS_LIGHT}}} {":root{" + TOKENS_DARK + "}" if dark else ""} body{{margin:0;padding:24px 16px;background:var(--color-bg);font-family:var(--font-anthropic-sans);}}</style>'
            f'{mock}</head><body><div id="dash-root">')
    # the Data type reruns every drawing when the page's width changes; rerun once the window has its final size
    tail = '</div><script>dash._run(); setTimeout(function () { dash._run(); }, 1500);</script></body></html>'
    return head + page + tail


def shot(url, png, size):
    subprocess.run(["timeout", "90", "google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars", "--virtual-time-budget=8000",
                    f"--window-size={size}", f"--screenshot={png}", url], capture_output=True)
    print(png, os.path.exists(png))


for dark in (False, True):
    mode = "dark" if dark else "light"
    p = os.path.join(HERE, f"harness_{mode}.html")
    open(p, "w").write(build(dark))
    shot(f"file://{p}", os.path.join(HERE, f"shot_{mode}_wide.png"), "1150,7600")
    # headless Chrome lays a window out at 500 px at least, so a phone is a 375 px frame in a 500 px window, cropped afterwards
    frame = os.path.join(HERE, f"harness_{mode}_phone.html")
    open(frame, "w").write(f'<!doctype html><html><body style="margin:0"><iframe src="harness_{mode}.html" '
                           f'style="width:375px;height:11500px;border:0;display:block"></iframe></body></html>')
    png = os.path.join(HERE, f"shot_{mode}_phone.png")
    shot(f"file://{frame}", png, "500,11500")
    try:
        from PIL import Image
        Image.open(png).crop((0, 0, 375, 11500)).save(png)
    except Exception as e:  # keep the uncropped shot
        print("crop skipped:", e)
