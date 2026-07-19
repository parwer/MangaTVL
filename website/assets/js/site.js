/* Shared site behaviour: mobile sidebar toggle. Page-specific
   visualizations live inline in each page (kept self-contained so the
   site works opened directly from disk, no server needed). */
(function () {
  function ready(fn){ if(document.readyState!=="loading"){fn();}else{document.addEventListener("DOMContentLoaded",fn);} }
  ready(function () {
    var btn = document.querySelector(".menu-btn");
    var sb  = document.querySelector(".sidebar");
    if (btn && sb) {
      btn.addEventListener("click", function () { sb.classList.toggle("open"); });
      document.querySelectorAll(".sidebar a").forEach(function (a) {
        a.addEventListener("click", function () { sb.classList.remove("open"); });
      });
    }
  });
})();

/* Tiny helper used by step-through visualizations on several pages.
   Wire up: a container with [data-stepper], buttons [data-step-prev/next],
   step dots [data-dots], and elements [data-step="N"] shown one at a time. */
function makeStepper(root) {
  var steps = Array.prototype.slice.call(root.querySelectorAll("[data-step]"));
  var dotsBox = root.querySelector("[data-dots]");
  var prev = root.querySelector("[data-step-prev]");
  var next = root.querySelector("[data-step-next]");
  var capBox = root.querySelector("[data-step-caption]");
  var captions = (root.getAttribute("data-captions") || "").split("||");
  var i = 0, n = steps.length;

  if (dotsBox) {
    dotsBox.innerHTML = "";
    for (var k = 0; k < n; k++) {
      var d = document.createElement("span");
      d.className = "d"; dotsBox.appendChild(d);
    }
  }
  function show(idx) {
    i = (idx + n) % n;
    steps.forEach(function (s, si) { s.style.display = (si === i) ? "" : "none"; });
    if (dotsBox) Array.prototype.forEach.call(dotsBox.children, function (d, di) { d.classList.toggle("on", di === i); });
    if (capBox && captions[i] !== undefined) capBox.innerHTML = captions[i];
  }
  if (prev) prev.addEventListener("click", function () { show(i - 1); });
  if (next) next.addEventListener("click", function () { show(i + 1); });
  show(0);
  return { show: show };
}

document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("[data-stepper]").forEach(makeStepper);
});

/* Draggable before/after image comparison. Wire up:
   <div class="img-compare" data-compare> with .ic-base img, .ic-top img, .ic-handle */
function makeCompare(el) {
  function setPos(clientX) {
    var r = el.getBoundingClientRect();
    var p = (clientX - r.left) / r.width * 100;
    p = Math.max(0, Math.min(100, p));
    el.style.setProperty("--pos", p + "%");
  }
  var dragging = false;
  el.addEventListener("pointerdown", function (e) { dragging = true; el.setPointerCapture(e.pointerId); setPos(e.clientX); });
  el.addEventListener("pointermove", function (e) { if (dragging) setPos(e.clientX); });
  el.addEventListener("pointerup",   function () { dragging = false; });
  el.addEventListener("pointercancel", function () { dragging = false; });
}
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("[data-compare]").forEach(makeCompare);
});
