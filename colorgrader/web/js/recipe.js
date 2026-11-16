import { html } from '/static/vendor/htm-preact-standalone.js';

// Stage 3 fills these panels: the editor recipe, LUT export and graded clip export.
export function RecipePanel() {
  return html`<div class="pad dim">The editor recipe arrives with the next build stage.</div>`;
}

export function ExportPanel() {
  return html`<div class="pad dim">LUT and clip export arrive with the next build stage.</div>`;
}
