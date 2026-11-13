"""
ui_theme.py - Modern dark "glass" design system + landing page for the Gradio app.
Style language: near-black #05070d, white/15 borders, cyan -> violet gradients,
glassmorphism, Inter font, sticky stacked cards, animated counters.
Only presentation lives here; all numbers shown come from results/*.json.
"""
from html import escape

# ------------------------------------------------------------------ <head>
HEAD = """
<link rel="icon" type="image/svg+xml" href="favicon.svg">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script>
(function () {
  // Counters: 0 -> target, requestAnimationFrame + cubic ease-out
  function runCounter(el) {
    if (el.dataset.done) return; el.dataset.done = "1";
    var target = parseFloat(el.dataset.count), dec = parseInt(el.dataset.dec || "0", 10);
    var suffix = el.dataset.suffix || "", dur = 2200, t0 = null;
    function step(ts) {
      if (t0 === null) t0 = ts;
      var p = Math.min((ts - t0) / dur, 1), e = 1 - Math.pow(1 - p, 3);
      el.textContent = (target * e).toFixed(dec) + suffix;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  // Stacked cards: scale down + dim as later sticky cards overlap them
  function stackFx() {
    var cards = document.querySelectorAll(".nx-stack-card");
    cards.forEach(function (c, i) {
      var next = cards[i + 1]; if (!next) return;
      var r = next.getBoundingClientRect(), mine = c.getBoundingClientRect();
      var overlap = Math.max(0, Math.min(1, (mine.bottom - r.top) / mine.height));
      c.style.transform = "scale(" + (1 - overlap * 0.06) + ")";
      c.style.filter = "brightness(" + (1 - overlap * 0.45) + ")";
    });
  }
  function scan() {
    var io = window.__nxIO = window.__nxIO || new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) {
          if (e.target.dataset.count) runCounter(e.target);
          e.target.classList.add("nx-in");
        }
      });
    }, { threshold: 0.2 });
    document.querySelectorAll("[data-count],.nx-reveal").forEach(function (el) {
      if (!el.dataset.obs) { el.dataset.obs = "1"; io.observe(el); }
    });
  }
  window.nxGo = function (idx) {            // jump to a top-level tab
    var tabs = document.querySelectorAll('.nx-tabs-root > .tab-wrapper button[role="tab"], .nx-tabs-root > div button[role="tab"]');
    if (tabs[idx]) { tabs[idx].click(); window.scrollTo({ top: 0, behavior: "smooth" }); }
  };
  window.addEventListener("scroll", stackFx, { passive: true });
  new MutationObserver(scan).observe(document.documentElement, { childList: true, subtree: true });
  window.addEventListener("load", scan);
})();
</script>
"""

# ------------------------------------------------------------------ CSS
CSS = """
:root{
  --nx-bg:#05070d; --nx-panel:rgba(255,255,255,.04); --nx-line:rgba(255,255,255,.15);
  --nx-text:#e6edf7; --nx-mute:#8b97ad; --nx-cyan:#67e8f9; --nx-violet:#8b5cf6; --nx-blue:#38bdf8;
}
html,body{background:var(--nx-bg)!important;}
*{scrollbar-width:none;} *::-webkit-scrollbar{display:none;}
.gradio-container{
  font-family:'Inter',system-ui,sans-serif!important; -webkit-font-smoothing:antialiased;
  background:
    radial-gradient(900px 500px at 15% -5%,rgba(56,189,248,.14),transparent 60%),
    radial-gradient(800px 500px at 90% 0%,rgba(139,92,246,.14),transparent 60%),
    var(--nx-bg)!important;
  max-width:100%!important; width:100%!important; margin:0; padding:0 clamp(16px,3vw,56px)!important; color:var(--nx-text); min-height:100vh; font-size:16px;
  --body-background-fill:#05070d; --background-fill-primary:rgba(255,255,255,.035);
  --background-fill-secondary:rgba(255,255,255,.03); --block-background-fill:rgba(255,255,255,.035);
  --block-border-color:rgba(255,255,255,.15); --border-color-primary:rgba(255,255,255,.15);
  --body-text-color:#e6edf7; --body-text-color-subdued:#8b97ad; --block-label-text-color:#8b97ad;
  --block-title-text-color:#c8d3e6; --input-background-fill:rgba(255,255,255,.04);
  --input-border-color:rgba(255,255,255,.15); --input-border-color-focus:rgba(103,232,249,.6);
  --table-even-background-fill:rgba(255,255,255,.02); --table-odd-background-fill:rgba(255,255,255,.045);
  --color-accent:#67e8f9; --slider-color:#67e8f9; --radius-lg:16px; --block-radius:16px;
}
.gradio-container .block,.gradio-container .form,.gradio-container .gr-group{
  backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
  border-radius:16px!important; box-shadow:inset 0 1px 0 rgba(255,255,255,.06);
}
.gradio-container h1,.gradio-container h2,.gradio-container h3{letter-spacing:-.02em;color:#fff}
.gradio-container .prose{color:#c8d3e6!important}
.gradio-container img{border-radius:14px}

/* top-level tabs as pill nav */
.nx-tabs-root>div>.tab-wrapper,.nx-tabs-root>.tab-wrapper{
  position:sticky; top:8px; z-index:50; padding:6px; margin:0 0 18px;
  border:1px solid var(--nx-line)!important; border-radius:999px;
  background:rgba(5,7,13,.65); backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
}
.gradio-container button[role="tab"]{
  border-radius:999px!important; border:none!important; color:var(--nx-mute)!important;
  font-weight:500; transition:background .3s,color .3s;
}
.gradio-container button[role="tab"]:hover{background:rgba(255,255,255,.1)!important;color:#fff!important}
.gradio-container button[role="tab"].selected{
  background:rgba(255,255,255,.12)!important;color:#fff!important;
  box-shadow:inset 0 0 0 1px rgba(103,232,249,.35);
}
.gradio-container button[role="tab"].selected::after{display:none!important}

/* buttons */
.gradio-container button.primary{
  position:relative; border:1px solid rgba(165,243,252,.7)!important; color:#fff!important; font-weight:600;
  background:linear-gradient(180deg,#0b1224,#060b18)!important; border-radius:999px!important;
  box-shadow:0 0 0 1px rgba(255,255,255,.06) inset,0 0 24px rgba(34,211,238,.35),
             0 0 48px rgba(59,130,246,.25),0 0 72px rgba(139,92,246,.2);
  transition:transform .3s,box-shadow .3s;
}
.gradio-container button.primary:hover{transform:scale(1.03);
  box-shadow:0 0 0 1px rgba(255,255,255,.1) inset,0 0 32px rgba(34,211,238,.5),
             0 0 64px rgba(59,130,246,.35),0 0 96px rgba(139,92,246,.3)}
.gradio-container button.secondary{border-radius:999px!important;border:1px solid var(--nx-line)!important;
  background:rgba(255,255,255,.05)!important;color:#fff!important;transition:transform .3s}
.gradio-container button.secondary:hover{transform:scale(1.02);background:rgba(255,255,255,.1)!important}
.gradio-container table{border-radius:12px;overflow:hidden}
.gradio-container footer{display:none!important}

/* ---------------- landing ---------------- */
.nx{font-family:'Inter',sans-serif;color:var(--nx-text)}
.nx *{box-sizing:border-box}
.nx-grad{background:linear-gradient(90deg,#67e8f9,#a78bfa);-webkit-background-clip:text;background-clip:text;color:transparent}
.nx-hero{position:relative;text-align:center;min-height:calc(100vh - 90px);padding:56px 20px 40px;overflow:hidden}
.nx-hero::before{content:"";position:absolute;inset:-40% -10% auto -10%;height:520px;z-index:0;
  background:radial-gradient(closest-side,rgba(56,189,248,.22),transparent),
             radial-gradient(closest-side at 70% 40%,rgba(139,92,246,.22),transparent);filter:blur(30px);
  animation:nxFloat 9s ease-in-out infinite alternate}
@keyframes nxFloat{to{transform:translate3d(30px,20px,0) scale(1.08)}}
.nx-hero>*{position:relative;z-index:1}
.nx-tag{position:relative;display:inline-flex;align-items:center;gap:8px;padding:7px 16px;border-radius:999px;
  font-size:13px;font-weight:500;color:#cffafe;border:1px solid rgba(165,243,252,.35);
  background:rgba(255,255,255,.05);backdrop-filter:blur(12px);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.2),0 0 28px rgba(34,211,238,.25)}
.nx-tag i{width:7px;height:7px;border-radius:50%;background:#67e8f9;box-shadow:0 0 10px #67e8f9;animation:nxPulse 2s infinite}
@keyframes nxPulse{50%{opacity:.35;transform:scale(.7)}}
.nx-h1{font-size:clamp(38px,6vw,72px);line-height:1.05;font-weight:700;letter-spacing:-.04em;color:#fff;margin:22px auto 0;max-width:900px}
.nx-sub{color:var(--nx-mute);font-size:19px;line-height:1.6;max-width:840px;margin:20px auto 0}
.nx-cta{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:32px}
.nx-btn{position:relative;display:inline-flex;align-items:center;gap:8px;padding:13px 26px;border-radius:999px;
  font-weight:600;font-size:15px;color:#fff;cursor:pointer;border:1px solid rgba(165,243,252,.7);
  background:linear-gradient(180deg,#0b1224,#050914);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.18),0 0 24px rgba(34,211,238,.4),0 0 56px rgba(59,130,246,.28),0 0 90px rgba(139,92,246,.22);
  transition:transform .3s,box-shadow .3s;text-decoration:none}
.nx-btn:hover{transform:scale(1.03)}
.nx-btn.ghost{border-color:var(--nx-line);background:rgba(255,255,255,.05);box-shadow:none}
.nx-btn.ghost:hover{background:rgba(255,255,255,.1)}

.nx-dash{margin:36px auto 0;width:100%;max-width:none;padding:6px;border-radius:26px;text-align:left;
  border:1px solid var(--nx-line);background:rgba(255,255,255,.04);backdrop-filter:blur(14px);
  box-shadow:0 30px 80px rgba(0,0,0,.5),0 0 80px rgba(56,189,248,.08)}
.nx-dash-in{border-radius:20px;background:#06111f;padding:18px;border:1px solid rgba(255,255,255,.08)}
.nx-dots{display:flex;gap:6px;margin-bottom:14px}.nx-dots b{width:10px;height:10px;border-radius:50%;background:rgba(255,255,255,.18)}
.nx-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.nx-metric{padding:14px;border-radius:14px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);
  transition:transform .3s,border-color .3s}
.nx-metric:hover{transform:translateY(-2px);border-color:rgba(103,232,249,.3)}
.nx-metric small{color:var(--nx-mute);font-size:12px}
.nx-metric .v{font-size:28px;font-weight:700;color:#fff;letter-spacing:-.03em;margin-top:4px}
.nx-metric .d{font-size:11px;color:#6ee7b7;margin-top:2px}
.nx-chart{margin-top:10px;padding:14px;border-radius:14px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.1)}
.nx-chart svg{width:100%;height:120px;display:block}
.nx-line{stroke-dasharray:900;stroke-dashoffset:900;animation:nxDraw 6s ease-in-out infinite}
@keyframes nxDraw{0%{stroke-dashoffset:900}60%,90%{stroke-dashoffset:0}100%{stroke-dashoffset:0;opacity:.3}}
.nx-bars{display:grid;gap:9px;margin-top:6px}
.nx-bar{display:grid;grid-template-columns:150px 1fr 52px;align-items:center;gap:10px;font-size:12px;color:var(--nx-mute)}
.nx-bar span.t{height:7px;border-radius:9px;background:rgba(255,255,255,.08);overflow:hidden}
.nx-bar span.t i{display:block;height:100%;border-radius:9px;background:linear-gradient(90deg,#22d3ee,#8b5cf6);
  width:0;transition:width 1.6s cubic-bezier(.2,.8,.2,1)}
.nx-in .nx-bar span.t i,.nx-bar.nx-in span.t i{width:var(--w)}

.nx-sec{padding:70px 20px 10px}
.nx-sec h2{font-size:clamp(28px,4vw,44px);font-weight:700;letter-spacing:-.03em;text-align:center;margin:0}
.nx-sec p.l{color:var(--nx-mute);text-align:center;max-width:720px;margin:12px auto 36px}
.nx-grid{display:grid;grid-template-columns:.85fr .85fr 1.3fr;gap:14px}
.nx-card{padding:22px;border-radius:20px;border:1px solid var(--nx-line);background:rgba(255,255,255,.04);
  backdrop-filter:blur(12px);transition:transform .3s,border-color .3s}
.nx-card:hover{transform:translateY(-3px);border-color:rgba(103,232,249,.3)}
.nx-card h3{margin:12px 0 6px;font-size:18px;color:#fff}.nx-card p{margin:0;color:var(--nx-mute);font-size:14px;line-height:1.6}
.nx-ico{width:40px;height:40px;border-radius:12px;display:grid;place-items:center;font-size:20px;
  background:linear-gradient(135deg,rgba(34,211,238,.2),rgba(139,92,246,.2));border:1px solid rgba(255,255,255,.15)}
.nx-badges{display:flex;gap:6px;flex-wrap:wrap;margin-top:14px}
.nx-badges span{font-size:11px;padding:4px 10px;border-radius:999px;border:1px solid rgba(255,255,255,.15);color:#cbd5e1;background:rgba(255,255,255,.05)}

.nx-stack{padding:0 0 60px}
.nx-stack-card{position:sticky;margin-bottom:26px;min-height:300px;padding:30px;border-radius:28px;
  border:1px solid var(--nx-line);overflow:hidden;transform-origin:center top;
  backdrop-filter:blur(14px);will-change:transform,filter;box-shadow:0 20px 60px rgba(0,0,0,.45)}
.nx-stack-card:nth-child(1){top:96px;z-index:1;background:radial-gradient(600px 300px at 90% 0%,rgba(45,212,191,.28),transparent),#071018}
.nx-stack-card:nth-child(2){top:112px;z-index:2;background:radial-gradient(600px 300px at 90% 0%,rgba(34,211,238,.28),transparent),#071019}
.nx-stack-card:nth-child(3){top:128px;z-index:3;background:radial-gradient(600px 300px at 90% 0%,rgba(59,130,246,.3),transparent),#070e1d}
.nx-stack-card:nth-child(4){top:144px;z-index:4;background:radial-gradient(600px 300px at 90% 0%,rgba(139,92,246,.32),transparent),#0a0a1c}
.nx-step{font-size:13px;font-weight:600;color:#67e8f9;letter-spacing:.12em}
.nx-stack-card h3{font-size:28px;margin:8px 0 8px;color:#fff;letter-spacing:-.03em}
.nx-stack-card p{color:#aab6cb;width:100%;line-height:1.65;margin:0}
.nx-kpis{display:flex;gap:12px;flex-wrap:wrap;margin-top:22px}
.nx-kpi{padding:12px 18px;border-radius:16px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.14);min-width:120px}
.nx-kpi b{display:block;font-size:26px;color:#fff;letter-spacing:-.03em}.nx-kpi small{color:var(--nx-mute);font-size:12px}
.nx-open{margin-top:20px;display:inline-flex;align-items:center;gap:8px;font-size:14px;font-weight:600;color:#a5f3fc;cursor:pointer}
.nx-pulse{display:inline-block;width:9px;height:9px;border-radius:50%;background:#a78bfa;box-shadow:0 0 0 0 rgba(167,139,250,.7);animation:nxRing 2s infinite;margin-right:8px}
@keyframes nxRing{to{box-shadow:0 0 0 14px rgba(167,139,250,0)}}
.nx-foot{text-align:center;color:var(--nx-mute);font-size:13px;padding:30px 0 10px;border-top:1px solid var(--nx-line);margin-top:20px}

.nx-reveal{opacity:0;transform:translateY(22px);transition:opacity .8s,transform .8s}
.nx-reveal.nx-in{opacity:1;transform:none}
.nx-header{display:flex;align-items:center;gap:12px;padding:6px 4px 4px}
.nx-header svg{height:30px}
.nx-header .who{margin-left:auto;color:var(--nx-mute);font-size:13px}

@media(max-width:1024px){.nx-grid{grid-template-columns:1fr 1fr}.nx-grid>:last-child{grid-column:1/-1}}
@media(max-width:768px){.nx-grid{grid-template-columns:1fr}.nx-metrics{grid-template-columns:1fr 1fr}
  .nx-bar{grid-template-columns:100px 1fr 46px}.nx-stack-card{padding:22px}}
.nx-fs{margin-left:12px;padding:7px 16px;border-radius:999px;border:1px solid var(--nx-line);background:rgba(255,255,255,.06);
  color:#fff;font-size:13px;font-weight:500;cursor:pointer;transition:transform .3s,background .3s}
.nx-fs:hover{transform:scale(1.03);background:rgba(255,255,255,.12)}
.gradio-container .prose{font-size:17px!important;line-height:1.7}
.gradio-container .block img{background:#fff;padding:6px}

/* force true full width (override Gradio's centered/max-width wrappers) */
.gradio-container,.gradio-container>.main,.gradio-container .main .wrap,.gradio-container .app,
.gradio-container .contain,.gradio-container .fillable{max-width:none!important;width:100%!important}
.gradio-container .contain{padding:0!important}
/* spacing: roomier buttons / rows / tabs */
.gradio-container button.primary,.gradio-container button.secondary,.gradio-container button.lg,.gradio-container button.sm{
  padding:12px 28px!important;min-height:46px;line-height:1.2}
.gradio-container .row,.gradio-container .form{gap:16px!important}
.gradio-container .column{gap:16px!important}
.gradio-container .block{padding:18px!important}
.gradio-container button[role="tab"]{padding:10px 20px!important;margin:0 3px}
.gradio-container .tab-wrapper,.gradio-container .tab-container{gap:6px}
.gradio-container .examples button,.gradio-container .gallery-item,.gradio-container table.table td{padding:10px 14px!important}
.nx-cta{gap:16px;margin-top:36px}.nx-btn{padding:15px 32px}
.nx-members{display:flex;flex-wrap:wrap;gap:8px;margin-left:12px}
.nx-member{font-size:12px;padding:6px 12px;border-radius:999px;border:1px solid var(--nx-line);background:rgba(255,255,255,.05);color:var(--nx-mute)}
.nx-member b{color:#fff;font-weight:600;margin-right:4px}
.nx-header{flex-wrap:wrap;padding:12px 4px 14px;gap:10px}
.nx-open{padding:10px 18px;border:1px solid rgba(165,243,252,.35);border-radius:999px;background:rgba(255,255,255,.05)}

/* ---------- tab bars: restyled (all levels) ---------- */
.gradio-container [role="tablist"],.gradio-container .tab-container{
  display:flex!important;flex-wrap:wrap!important;gap:8px!important;height:auto!important;
  padding:8px!important;border:1px solid var(--nx-line)!important;border-radius:22px!important;
  background:rgba(255,255,255,.04)!important;backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.08)!important;margin-bottom:18px}
.gradio-container [role="tablist"]::after,.gradio-container .tab-container::after{display:none!important;content:none!important}
.gradio-container .tab-wrapper{height:auto!important;padding:0!important;border:none!important;background:none!important}
.gradio-container button[role="tab"]{
  flex:1 1 auto;height:auto!important;font-size:15px!important;font-weight:500!important;letter-spacing:-.01em;
  padding:13px 24px!important;margin:0!important;border-radius:16px!important;border:1px solid transparent!important;
  color:#9aa7bd!important;background:transparent!important;text-align:center;white-space:nowrap}
.gradio-container button[role="tab"]:hover{color:#fff!important;background:rgba(255,255,255,.08)!important}
.gradio-container button[role="tab"].selected,.gradio-container button[role="tab"][aria-selected="true"]{
  color:#fff!important;font-weight:600!important;
  background:linear-gradient(135deg,rgba(34,211,238,.22),rgba(139,92,246,.26))!important;
  border-color:rgba(165,243,252,.45)!important;
  box-shadow:0 0 22px rgba(34,211,238,.25),inset 0 1px 0 rgba(255,255,255,.2)!important}
/* top-level bar: sticky, larger */
.gradio-container .nx-tabs-root>.tab-wrapper,.gradio-container .nx-tabs-root>div>.tab-wrapper{
  position:sticky;top:10px;z-index:60}
.gradio-container .nx-tabs-root>.tab-wrapper [role="tablist"],.gradio-container .nx-tabs-root>div>.tab-wrapper [role="tablist"],
.gradio-container .nx-tabs-root>.tab-wrapper .tab-container{
  background:rgba(5,7,13,.78)!important;padding:10px!important;border-radius:26px!important;
  box-shadow:0 12px 40px rgba(0,0,0,.5),inset 0 1px 0 rgba(255,255,255,.1)!important}
.gradio-container .nx-tabs-root>.tab-wrapper button[role="tab"],.gradio-container .nx-tabs-root>div>.tab-wrapper button[role="tab"]{
  font-size:16px!important;padding:16px 28px!important}
/* inner tab bars: compact secondary look */
.gradio-container .tabitem [role="tablist"],.gradio-container .tabitem .tab-container{background:transparent!important;border-radius:16px!important;box-shadow:none!important}
.gradio-container .tabitem button[role="tab"]{font-size:14px!important;padding:10px 20px!important;flex:0 1 auto}
.gradio-container .tabitem{padding:0!important;border:none!important}
"""


# ------------------------------------------------------------------ pieces
def header_html(title, members):
    logo = (
        '<svg viewBox="0 0 150 40" xmlns="http://www.w3.org/2000/svg" aria-label="AI Final Project">'
        '<defs><linearGradient id="lg" x1="0" x2="1"><stop offset="0" stop-color="#67e8f9"/>'
        '<stop offset="1" stop-color="#a78bfa"/></linearGradient></defs>'
        '<text x="0" y="30" font-family="Inter,sans-serif" font-weight="700" font-size="28" '
        'letter-spacing="-1" fill="url(#lg)">AI Final</text></svg>'
    )
    chips = "".join(f'<span class="nx-member"><b>{escape(n)}</b> {escape(i)}</span>' for n, i in members)
    return (f'<div class="nx nx-header">{logo}<div class="nx-members">{chips}</div>'
            '<span class="who">Informatics &middot; Semester 20261</span>'
            '<button class="nx-fs" onclick="document.fullscreenElement?document.exitFullscreen():'
            'document.documentElement.requestFullscreen()">&#x26F6; Full screen</button></div>')


def landing_html(m):
    """m: dict of real metrics (see app.landing_metrics)."""
    def n(key, default=0.0):
        return m.get(key, default)

    spam, grad, seg, wx = n("spam_acc"), n("grad_acc"), n("segments", 3), n("weather_acc")
    bars = [
        ("Spam - Naive Bayes", spam), ("Graduation - Perceptron", grad),
        ("Graduation - k-NN", n("knn_acc")), ("Weather - Neural Net", wx),
    ]
    bar_html = "".join(
        f'<div class="nx-bar nx-reveal"><span>{escape(l)}</span><span class="t"><i style="--w:{v:.1f}%"></i></span>'
        f'<span>{v:.1f}%</span></div>' for l, v in bars)

    def metric(label, value, dec, suffix, note):
        return (f'<div class="nx-metric"><small>{label}</small>'
                f'<div class="v" data-count="{value}" data-dec="{dec}" data-suffix="{suffix}">0{suffix}</div>'
                f'<div class="d">{note}</div></div>')

    metrics = (metric("Spam accuracy", f"{spam:.2f}", 2, "%", "Naive Bayes")
               + metric("Graduation accuracy", f"{grad:.1f}", 1, "%", "Averaged Perceptron")
               + metric("Customer segments", f"{seg}", 0, "", "K-Means")
               + metric("Weather accuracy", f"{wx:.2f}", 2, "%", "12 neural nets compared"))

    def card(step, title, text, kpis, tab):
        k = "".join(f'<div class="nx-kpi"><b>{a}</b><small>{b}</small></div>' for a, b in kpis)
        return (f'<div class="nx-stack-card"><div class="nx-step">{step}</div><h3>{title}</h3><p>{text}</p>'
                f'<div class="nx-kpis">{k}</div>'
                f'<div class="nx-open" onclick="nxGo({tab})">Open demo &rarr;</div></div>')

    stack = (
        card("01 &middot; QUESTION 1", "Spam Classifier",
             "Naive Bayes with word + character TF-IDF, tuned by 5-fold CV. Includes confusion matrix, "
             "calibration and an uncertainty analysis showing where the model is overconfident.",
             [(f"{spam:.2f}%", "test accuracy"), (f"{n('spam_prec'):.1f}%", "precision"), (f"{n('spam_auc'):.3f}", "ROC-AUC")], 1)
        + card("02 &middot; QUESTION 2A", "Graduation Prediction",
               '<span class="nx-pulse"></span>Averaged Perceptron vs k-NN, compared with nested cross-validation. '
               "Live sliders for study hours, attendance and assignments.",
               [(f"{grad:.1f}%", "Perceptron test"), (f"{n('knn_acc'):.1f}%", "k-NN test"),
                (f"{n('grad_cv'):.1f}%", "nested CV")], 2)
        + card("03 &middot; QUESTION 2B", "Customer Segmentation",
               "K-Means on standardised age and monthly spending; k chosen with silhouette and "
               "Davies-Bouldin. Place a customer on the map and get a marketing suggestion.",
               [(f"{seg}", "clusters"), (f"{n('rows')}", "customers")], 3)
        + card("04 &middot; QUESTION 3", "Weather Neural Networks",
               "12 networks (1-3 hidden layers x 32-256 neurons) trained with Adam and early stopping. "
               "Compare architectures, learning curves and try real test-set examples.",
               [(f"{wx:.2f}%", "best test acc."), (f"{n('wx_layers')}x{n('wx_width')}", "best architecture"),
                ("12", "models")], 4)
    )

    return f"""
<div class="nx">
  <section class="nx-hero">
    <h1 class="nx-h1">Four AI models, one <span class="nx-grad">intelligent</span> workspace</h1>
    <p class="nx-sub">Classify spam, predict graduation, segment customers and classify weather. Every model is trained,
      evaluated and ready to try right in your browser.</p>
    <div class="nx-cta">
      <a class="nx-btn" onclick="nxGo(1)">Try it now &rarr;</a>
      <a class="nx-btn ghost" onclick="document.getElementById('nx-how').scrollIntoView({{behavior:'smooth'}})">How it works</a>
    </div>
    <div class="nx-dash nx-reveal"><div class="nx-dash-in">
      <div class="nx-dots"><b></b><b></b><b></b></div>
      <div class="nx-metrics">{metrics}</div>
      <div class="nx-chart">
        <small style="color:#8b97ad;font-size:12px">Model accuracy overview</small>
        <svg viewBox="0 0 400 120" preserveAspectRatio="none">
          <defs><linearGradient id="nxa" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stop-color="#22d3ee" stop-opacity=".35"/><stop offset="1" stop-color="#22d3ee" stop-opacity="0"/>
          </linearGradient></defs>
          <path d="M0 100 L60 62 L120 76 L190 34 L260 48 L320 18 L400 26 L400 120 L0 120Z" fill="url(#nxa)"/>
          <path class="nx-line" d="M0 100 L60 62 L120 76 L190 34 L260 48 L320 18 L400 26" fill="none" stroke="#67e8f9" stroke-width="2.5" stroke-linecap="round"/>
        </svg>
      </div>
      <div class="nx-chart"><div class="nx-bars">{bar_html}</div></div>
    </div></div>
  </section>

  <section class="nx-sec">
    <h2 class="nx-reveal">Everything you need to <span class="nx-grad">evaluate</span> a model</h2>
    <p class="l">Each task ships with the same rigorous workflow.</p>
    <div class="nx-grid">
      <div class="nx-card nx-reveal"><div class="nx-ico">&#128274;</div><h3>Leak-free evaluation</h3>
        <p>Duplicates removed before splitting, tuning only on training data.</p>
        <div class="nx-badges"><span>Stratified</span><span>5-fold CV</span><span>Nested CV</span></div></div>
      <div class="nx-card nx-reveal"><div class="nx-ico">&#9889;</div><h3>Instant predictions</h3>
        <p>Pre-trained models load from disk; no retraining needed to demo.</p></div>
      <div class="nx-card nx-reveal"><div class="nx-ico">&#128202;</div><h3>Real-time insights</h3>
        <p>Confusion matrices, calibration, learning curves, silhouette scores and
        per-class reports for every question, with plain-language interpretation.</p>
        <div class="nx-badges"><span>Confusion matrix</span><span>ROC-AUC</span><span>Learning curves</span><span>Silhouette</span></div></div>
    </div>
  </section>

  <section class="nx-sec" id="nx-how">
    <h2 class="nx-reveal">Four questions, <span class="nx-grad">stacked</span></h2>
    <p class="l">Scroll to explore each task, then open its live demo.</p>
    <div class="nx-stack">{stack}</div>
  </section>
  <div class="nx-foot">Built with scikit-learn &amp; Gradio &middot; Informatics &middot; Semester 20261</div>
</div>
"""

