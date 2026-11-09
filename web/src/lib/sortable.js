// Click-to-sort for tables marked data-sortable. Header cells with data-type="num"
// sort numerically using each cell's data-v (falls back to text). Keyboard
// accessible because the sort control is a real <button>.
export function enableSorting(root = document) {
  root.querySelectorAll("table[data-sortable]").forEach((table) => {
    const heads = [...table.tHead.rows[0].cells];
    heads.forEach((th, col) => {
      if (th.dataset.nosort !== undefined) return;
      const label = th.textContent.trim();
      th.textContent = "";
      const b = document.createElement("button");
      b.type = "button"; b.className = "sort"; b.textContent = label;
      th.appendChild(b);
      b.addEventListener("click", () => {
        const dir = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
        heads.forEach((h) => h.removeAttribute("aria-sort"));
        th.setAttribute("aria-sort", dir);
        const num = th.dataset.type === "num";
        const body = table.tBodies[0];
        const rows = [...body.rows];
        const val = (r) => {
          const c = r.cells[col];
          const v = c.dataset.v ?? c.textContent.trim();
          return num ? (v === "" || v === "–" ? -Infinity : parseFloat(v)) : v.toLowerCase();
        };
        rows.sort((a, b2) => {
          const x = val(a), y = val(b2);
          const c = x < y ? -1 : x > y ? 1 : 0;
          return dir === "ascending" ? c : -c;
        });
        rows.forEach((r) => body.appendChild(r));
      });
    });
  });
}
