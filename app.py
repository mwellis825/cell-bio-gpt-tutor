
import os, re, json, pathlib, random, time
import streamlit as st

st.set_page_config(page_title="Let's Practice Biology!", page_icon="🎓", layout="wide")

# Hide "Press Enter to apply" text globally
st.markdown(
    """
    <style>
      div[data-testid="stWidgetLabelHelp"] {display: none !important;}
      div[role="alert"] p {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Placeholder minimal to demonstrate fix in HTML builder (removed buggy "+".join)
labels = ["Upstream factors","Core step","Intermediates/Evidence","Immediate output"]
terms  = ["enzyme","substrate","protein","product"]
answer = {t: labels[i%4] for i,t in enumerate(terms)}

items_html = "".join([f'<li class="card">{t}</li>' for t in terms])
bins_html = "".join([
    f"""
    <div class="bin">
      <div class="title">{lbl}</div>
      <ul id="bin_{i}" class="droplist"></ul>
    </div>
    """ for i,lbl in enumerate(labels)
])

html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js"></script>
  <style>
    .bank, .bin {{
      border: 2px dashed #bbb; border-radius: 10px; padding: 12px; min-height: 120px;
      background: #fafafa; margin-bottom: 14px;
    }}
    .bin {{ background: #f6faff; }}
    .droplist {{ list-style: none; margin: 0; padding: 0; min-height: 90px; }}
    .card {{
      background: white; border: 1px solid #ddd; border-radius: 8px;
      padding: 8px 10px; margin: 6px 0; cursor: grab;
      box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }}
    .zone {{ display:flex; gap:16px; }}
    .left {{ flex: 1; }}
    .right {{ flex: 2; display:grid; grid-template-columns: repeat(2, 1fr); gap:16px; }}
    .title {{ font-weight: 600; margin-bottom: 6px; }}
    .ok   {{ color:#0a7; font-weight:600; }}
    .bad  {{ color:#b00; font-weight:600; }}
    button {{
      border-radius: 8px; border: 1px solid #ddd; background:#fff; padding:8px 12px; cursor:pointer;
    }}
  </style>
</head>
<body>
  <div class="zone">
    <div class="left">
      <div class="title">Bank</div>
      <ul id="bank" class="bank droplist">{items_html}</ul>
    </div>
    <div class="right">{bins_html}</div>
  </div>
  <div style="margin-top:10px;">
    <button id="check">Check bins</button>
    <span id="score" style="margin-left:10px;"></span>
  </div>

  <script>
    const LABELS = {json.dumps(labels)};
    const ANSWERS = {json.dumps(answer)};

    const opts = {{
      group: {{ name: 'bins', pull: true, put: true }},
      animation: 150,
      forceFallback: true,
      fallbackOnBody: true,
      swapThreshold: 0.65,
      emptyInsertThreshold: 8,
    }};

    const lists = [document.getElementById('bank')];
    LABELS.forEach((_, i) => lists.push(document.getElementById('bin_'+i)));
    lists.forEach(el => new Sortable(el, opts));

    function readBins() {{
      const bins = {{}};
      LABELS.forEach((label, i) => {{
        const ul = document.getElementById('bin_'+i);
        const items = Array.from(ul.querySelectorAll('.card')).map(li => li.textContent.trim());
        bins[label] = items;
      }});
      return bins;
    }}

    document.getElementById('check').addEventListener('click', () => {{
      const bins = readBins();
      let total = 0, correct = 0;
      for (const [term, want] of Object.entries(ANSWERS)) {{
        total += 1;
        let got = "Bank";
        for (const [label, items] of Object.entries(bins)) {{
          if (items.includes(term)) {{ got = label; break; }}
        }}
        if (got === want) correct += 1;
      }}
      const score = document.getElementById('score');
      if (total === 0) {{
        score.innerHTML = "<span class='bad'>Drag terms into bins first.</span>";
      }} else if (correct === total) {{
        score.innerHTML = "<span class='ok'>All bins correct! 🎉</span>";
      }} else {{
        score.innerHTML = "<span class='bad'>" + correct + "/" + total + " correct — try again.</span>";
      }}
    }});
  </script>
</body>
</html>
"""

st.title("Let's Practice Biology!")
st.components.v1.html(html, height=640, scrolling=True)
