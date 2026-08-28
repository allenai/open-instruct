const state = {
  meta: null,
  steps: [],
  run: null,
  step: null,
  source: "accepted",
  mode: "trajectories",
  category: "review",
  groupKey: "",
  sort: "suspicion",
  search: "",
  stepsPayload: null,
  page: 0,
  hasMore: false,
  loading: false,
  requestVersion: 0,
  categories: [],
  groupCategories: [],
  categoryCounts: {},
  sourceRanges: {},
  detailId: null,
  traceOffset: 0,
  groupModalKey: null,
  groupModalRequestVersion: 0,
  inspectorInitialized: false,
};

const catalogState = {
  trainings: [],
  summary: {},
  refreshedAt: null,
  search: "",
  classification: "primary",
  sort: "recent",
  includeArchived: false,
  includeSmoke: false,
  loading: false,
  trainingId: null,
  refreshPoll: null,
  detailRequestVersion: 0,
  metricsByTraining: {},
};

const evaluationState = {
  trainingId: null,
  evaluationId: null,
  outcome: "all",
  sort: "id",
  search: "",
  page: 0,
  hasMore: false,
  loading: false,
  records: [],
  selectedQueryId: null,
  listRequestVersion: 0,
  detailRequestVersion: 0,
};

const fullDocumentState = {
  requestVersion: 0,
  payload: null,
  activeSource: "reference",
  referenceResult: null,
  searchResult: null,
  currentIndex: -1,
  searchDebounce: null,
  returnFocus: null,
};

const SEGMENT_LABELS = {
  prompt: "Question",
  reasoning: "Reasoning",
  assistant_text: "Assistant text",
  final_output: "Final response",
  tool_call: "Tool call",
  tool_result: "Tool result",
  user_text: "User turn",
};

// Tool observations are the bulk of a trajectory, so they start collapsed.
const COLLAPSED_KINDS = new Set(["tool_result", "prompt"]);

// Natural-language segments read better in the body font than in monospace.
const PROSE_KINDS = new Set(["reasoning", "assistant_text", "final_output", "prompt", "user_text"]);

// Keep the on-disk/API names stable, but describe the group-level fate in the UI.
// A retained group contains both reward-0 and reward-1 trajectories.
const SOURCE_LABELS = {
  accepted: "learner batch",
  filtered: "discarded group",
  both: "both sources",
};
const SOURCE_DESCRIPTIONS = {
  accepted: "Prompt group passed active sampling and entered the prepared learner batch; individual trajectories can still have reward 0 or be masked later.",
  filtered: "Prompt group had zero reward variance and was discarded before the learner update.",
  both: "Show retained learner-batch trajectories and saved active-sampling discards together.",
};
const BADGE_LABELS = { filtered: "discarded group" };

const OUTCOME_LABELS = {
  judged_correct: "verifier correct",
  judged_incorrect: "verifier incorrect",
  incomplete: "verifier skipped",
};
const GROUP_SHAPE_LABELS = {
  all_wrong_group: "Every sample of this prompt failed",
  all_correct_group: "Every sample of this prompt succeeded",
  mixed_group: "This prompt had both successes and failures",
};
const GROUP_DIFFICULTY_LABELS = {
  all_wrong_group: "All wrong · 0% pass rate",
  hard_group: "Hard · above 0% and at most 25%",
  learning_group: "Learning zone · above 25% and below 75%",
  easy_group: "Easy · at least 75% and below 100%",
  all_correct_group: "All correct · 100% pass rate",
};
// Shown elsewhere on the card, so they would only be noise as badges.
const HIDDEN_BADGES = new Set(["long", "mixed_group", "judged_correct", "judged_incorrect", "incomplete"]);

const sourceLabel = (source) => SOURCE_LABELS[source] || source;
const badgeLabel = (category) => BADGE_LABELS[category] || category.replaceAll("_", " ");
const outcomeLabel = (outcome) => OUTCOME_LABELS[outcome] || outcome || "unknown";

const elements = Object.fromEntries(
  [
    "catalog-view", "overview-panel", "training-panel", "evaluation-panel", "catalog-refreshed", "catalog-refresh",
    "catalog-summary", "catalog-search", "catalog-classification", "catalog-sort", "catalog-archived", "catalog-smoke",
    "catalog-count", "training-table", "catalog-empty", "training-breadcrumb", "training-detail",
    "rollout-view", "rollout-empty", "inspector-back",
    "evaluation-training-link", "evaluation-breadcrumb", "evaluation-header", "evaluation-stats",
    "evaluation-search", "evaluation-outcome", "evaluation-sort", "evaluation-result-count", "evaluation-result-caption",
    "evaluation-records", "evaluation-load-more", "evaluation-detail",
    "run-select", "root-path", "run-meta", "step-input", "step-range", "previous-step", "next-step",
    "step-slider", "step-caption", "step-latest", "view-mode-control", "view-mode-help",
    "source-control", "source-help", "category-list", "refresh-button", "last-refresh", "view-title", "view-subtitle",
    "search-input", "sort-select", "stat-records-label", "stat-records", "stat-reward-label", "stat-reward", "stat-tokens", "stat-validation", "stat-flagged",
    "result-count", "loading-label", "empty-state", "record-grid", "load-sentinel", "load-more",
    "group-modal-backdrop", "group-modal", "group-modal-close", "group-modal-kicker", "group-modal-title", "group-modal-summary", "group-modal-body",
    "drawer-backdrop", "detail-drawer", "drawer-close", "detail-copy-locator", "detail-kicker", "detail-title", "detail-body", "toast",
    "full-document-backdrop", "full-document-pane", "full-document-close", "full-document-title", "full-document-source",
    "full-document-toolbar", "full-document-body",
  ].map((id) => [id, document.getElementById(id)])
);

const categoryLabels = {
  all: "All trajectories",
  review: "Suspicious trajectories",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value, digits = 0) {
  if (value === null || value === undefined) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatTokens(value) {
  if (value === null || value === undefined) return "—";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}m`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { elements.toast.hidden = true; }, 5500);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload;
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleString();
}

function liveValidation(training) {
  return training.live?.validation || training.validation || {};
}

function liveRollouts(training) {
  return training.live?.rollouts || training.rollouts || {};
}

function rolloutAvailable(training) {
  const live = liveRollouts(training);
  if (typeof live.available === "boolean") return live.available;
  return (training.launches || []).some((launch) =>
    (launch.rollouts || []).some((rollout) => rollout.exists && (rollout.present_attempts || []).length)
  );
}

function checkpointStatus(checkpoint) {
  if (!checkpoint) return { label: "not recorded", available: false };
  if (checkpoint.complete) return { label: `step ${checkpoint.step}`, available: true };
  if (checkpoint.exists) return { label: `step ${checkpoint.step} · incomplete`, available: false };
  return { label: `step ${checkpoint.step} · missing`, available: false };
}

function validationScore(item) {
  if (!item) return null;
  for (const value of [item.score, item.value, item.accuracy]) {
    if (value !== null && value !== undefined && Number.isFinite(Number(value))) return Number(value);
  }
  return null;
}

function validationStep(item) {
  if (!item) return null;
  return item.optimizer_step ?? item.model_step ?? item.step ??
    (item.artifact_step === null || item.artifact_step === undefined ? null : Number(item.artifact_step) + 1);
}

function externalLink(url, label, className = "external-link") {
  if (!url) return "";
  return `<a class="${className}" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(label)} ↗</a>`;
}

function pathRow(label, path, { exists = null, url = null } = {}) {
  if (!path) return "";
  const status = exists === null ? "" : `<i class="status-dot ${exists ? "ready" : "missing"}" title="${exists ? "Available" : "Missing"}"></i>`;
  return `<div class="artifact-path">
    <span class="artifact-label">${escapeHtml(label)}</span>
    <div class="artifact-value">${status}<code title="${escapeHtml(path)}">${escapeHtml(path)}</code></div>
    <div class="artifact-actions">
      <button class="copy-button" data-copy="${escapeHtml(path)}" type="button">Copy</button>
      ${url ? externalLink(url, "Open", "path-link") : ""}
    </div>
  </div>`;
}

function tagPills(tags = {}) {
  return Object.entries(tags)
    .map(([key, value]) => `<span class="tag-pill" title="${escapeHtml(key)}">${escapeHtml(value)}</span>`)
    .join("");
}

function bestFullEval(training) {
  const best = training.best_evaluation;
  if (!best) return `<span class="metric-empty">Not evaluated</span>`;
  return `<strong>${formatPercent(best.score, 2)}</strong><small>${formatNumber(best.correct)}/${formatNumber(best.total)} · step ${formatNumber(best.step)}</small>`;
}

function bestEvaluationForFamily(training, family) {
  return fullEvaluations(training)
    .filter((item) => evaluationFamily(item.benchmark) === family)
    .reduce((winner, item) => {
      if (!winner || Number(item.score) > Number(winner.score)) return item;
      if (Number(item.score) === Number(winner.score) && Number(item.step) > Number(winner.step)) return item;
      return winner;
    }, null);
}

function bestFamilyEvaluation(training, family) {
  const best = bestEvaluationForFamily(training, family);
  if (!best) return `<span class="metric-empty">Not evaluated</span>`;
  return `<strong>${formatPercent(best.score, 2)}</strong><small>${formatNumber(best.correct)}/${formatNumber(best.total)} · step ${formatNumber(best.step)}</small>`;
}

function bestLiveValidation(training) {
  const validation = liveValidation(training);
  const best = validation.best || (validation.evaluations || []).reduce((winner, item) => {
    const score = validationScore(item);
    return !winner || score > validationScore(winner) ? item : winner;
  }, null);
  const score = validationScore(best);
  if (score === null) {
    const status = typeof validation.status === "string" ? validation.status : validation.status?.state;
    return `<span class="metric-empty">${escapeHtml(["loading", "pending", "refreshing"].includes(status) ? "Refreshing…" : "No score")}</span>`;
  }
  const step = validationStep(best);
  return `<strong>${formatPercent(score)}</strong><small>${step === null ? "best logged score" : `best · step ${formatNumber(step)}`}</small>`;
}

function fullEvaluations(training) {
  if (Array.isArray(training.evaluations) && training.evaluations.length) return training.evaluations;
  return training.best_evaluation ? [training.best_evaluation] : [];
}

function evaluationFamily(benchmark) {
  const normalized = String(benchmark || "").toLowerCase().replaceAll("_", "-");
  if (normalized.includes("terminal-bench") || normalized.includes("tb2")) return "browsecomp-plus";
  if (normalized.includes("tblite")) return "browsecomp";
  if (normalized.includes("browsecomp-plus")) return "browsecomp-plus";
  if (normalized.includes("browsecomp")) return "browsecomp";
  return "other";
}

function evaluationColumn(title, description, evaluations, trainingId) {
  const ordered = [...evaluations].sort((left, right) => Number(left.step) - Number(right.step));
  const best = ordered.reduce((winner, item) =>
    !winner || Number(item.score) > Number(winner.score) ? item : winner, null);
  const rows = ordered.map((item) => {
    const isBest = item === best;
    const checkpoint = checkpointStatus(item.checkpoint);
    const inspectable = Boolean(item.inference_artifact?.complete);
    const tag = inspectable ? "a" : "div";
    const href = inspectable
      ? ` href="#/trainings/${encodeURIComponent(trainingId)}/evaluations/${encodeURIComponent(item.id)}"`
      : "";
    return `<${tag} class="full-evaluation-row${inspectable ? " inspectable" : ""}"${href}>
      <div class="full-evaluation-step"><span>Step</span><strong>${formatNumber(item.step)}</strong></div>
      <div class="full-evaluation-score"><strong>${formatPercent(item.score)}</strong><span>${formatNumber(item.correct)}/${formatNumber(item.total)} correct</span></div>
      <div class="full-evaluation-source"><code title="${escapeHtml(item.benchmark)}">${escapeHtml(item.benchmark)}</code><span>${isBest ? '<b class="evaluation-best">Best</b>' : ""}${item.checkpoint ? `<i class="status-dot ${checkpoint.available ? "ready" : "missing"}" title="Checkpoint ${checkpoint.available ? "available" : "missing"}"></i>` : ""}${inspectable ? '<b class="evaluation-inspect">Inspect →</b>' : '<b class="evaluation-aggregate">Aggregate only</b>'}${item.beaker_url ? `<a class="evaluation-beaker" href="${escapeHtml(item.beaker_url)}" target="_blank" rel="noopener" title="Open the eval experiment on Beaker" onclick="event.stopPropagation()">Beaker ↗</a>` : ""}</span></div>
    </${tag}>`;
  }).join("");
  return `<article class="full-evaluation-column">
    <header><div><h3>${escapeHtml(title)}</h3><p>${escapeHtml(description)}</p></div><span>${formatNumber(ordered.length)} result${ordered.length === 1 ? "" : "s"}</span></header>
    <div class="full-evaluation-list">${rows || '<p class="full-evaluation-empty">No attributed evaluation results.</p>'}</div>
  </article>`;
}

function fullEvaluationSection(training) {
  const evaluations = fullEvaluations(training);
  if (!evaluations.length) return "";
  const plus = evaluations.filter((item) => evaluationFamily(item.benchmark) === "browsecomp-plus");
  const browsecomp = evaluations.filter((item) => evaluationFamily(item.benchmark) === "browsecomp");
  const other = evaluations.filter((item) => evaluationFamily(item.benchmark) === "other");
  return `<section class="full-evaluation-section" aria-labelledby="full-evaluation-title">
    <div class="section-title-row">
      <div><p class="eyebrow">External evaluation</p><h2 id="full-evaluation-title">Benchmark results by checkpoint</h2></div>
      <span class="muted">${formatNumber(evaluations.length)} recorded result${evaluations.length === 1 ? "" : "s"}</span>
    </div>
    <div class="full-evaluation-grid">
      ${evaluationColumn("Terminal-Bench 2.1", "Terminal-Bench 2.1 evaluations (89 tasks × 5 trials)", plus, training.id)}
      ${evaluationColumn("TBlite", "openthoughts-tblite 2.0 evaluations (100 tasks × 5 trials)", browsecomp, training.id)}
      ${other.length ? evaluationColumn("Other", "Other attributed full evaluations", other, training.id) : ""}
    </div>
  </section>`;
}

function classificationLabel(value) {
  return value === "smoke" ? "smoke / startup" : value || "unclassified";
}

function trainingActivity(training) {
  const liveUpdated = Number(training.live?.rollouts?.updated);
  if (Number.isFinite(liveUpdated) && liveUpdated > 0) return liveUpdated * 1000;
  const timestamps = (training.live?.rollouts?.attempt_metadata || [])
    .map((item) => new Date(item.timestamp || item.created_at || 0).getTime())
    .filter(Number.isFinite);
  return timestamps.length ? Math.max(...timestamps) : 0;
}

function comparableValidation(training) {
  const best = liveValidation(training).best;
  return validationScore(best) ?? training.best_evaluation?.score ?? -1;
}

function compareTrainings(left, right) {
  const titleOrder = String(left.title || left.id).localeCompare(String(right.title || right.id));
  if (catalogState.sort === "name") return titleOrder;
  if (catalogState.sort === "oldest") return trainingActivity(left) - trainingActivity(right) || titleOrder;
  if (catalogState.sort === "validation") {
    return comparableValidation(right) - comparableValidation(left) || titleOrder;
  }
  if (catalogState.sort === "step") {
    return Number(right.furthest_step ?? -1) - Number(left.furthest_step ?? -1) || titleOrder;
  }
  return trainingActivity(right) - trainingActivity(left) || titleOrder;
}

function filteredTrainings() {
  const query = catalogState.search.toLowerCase();
  return catalogState.trainings.filter((training) => {
    if (training.visibility === "hidden" && !catalogState.includeSmoke) return false;
    if (training.visibility === "archive" && !catalogState.includeArchived) return false;
    if (training.classification === "smoke" && !catalogState.includeSmoke) return false;
    if (
      catalogState.classification === "primary"
      && !["substantive", "evaluated"].includes(training.classification)
      && !(catalogState.includeSmoke && training.classification === "smoke")
    ) return false;
    if (catalogState.classification !== "primary" && training.classification !== catalogState.classification) return false;
    if (!query) return true;
    const haystack = [
      training.id,
      training.title,
      training.classification,
      training.wandb?.run_id,
      ...Object.keys(training.tags || {}),
      ...Object.values(training.tags || {}),
    ].filter(Boolean).join(" ").toLowerCase();
    return haystack.includes(query);
  }).sort(compareTrainings);
}

function renderCatalogSummary() {
  const trainings = catalogState.trainings;
  const counts = {
    total: trainings.length,
    evaluated: trainings.filter((item) => item.classification === "evaluated").length,
    substantive: trainings.filter((item) => item.classification === "substantive").length,
    rollouts: trainings.filter(rolloutAvailable).length,
  };
  elements["catalog-summary"].innerHTML = [
    [counts.total, "registered"],
    [counts.evaluated, "fully evaluated"],
    [counts.substantive, "substantive"],
    [counts.rollouts, "with rollouts"],
  ].map(([value, label]) => `<div><strong>${formatNumber(value)}</strong><span>${escapeHtml(label)}</span></div>`).join("");
}

function updateRefreshProgress() {
  const refresh = catalogState.summary?.validation_refresh;
  if (!refresh || !["pending", "refreshing"].includes(refresh.state)) {
    clearTimeout(catalogState.refreshPoll);
    catalogState.refreshPoll = null;
    elements["catalog-refreshed"].textContent = catalogState.refreshedAt
      ? `Checked ${formatDate(catalogState.refreshedAt)}`
      : "";
    return;
  }
  const progress = refresh.total
    ? ` · ${formatNumber(refresh.completed)}/${formatNumber(refresh.total)}`
    : "";
  elements["catalog-refreshed"].textContent = `Refreshing W&B${progress}`;
  clearTimeout(catalogState.refreshPoll);
  catalogState.refreshPoll = setTimeout(async () => {
    const route = parseRoute();
    try {
      if (route.view === "overview") await loadCatalog({ force: true, quiet: true });
      else await loadTrainingDetail(route.trainingId, { quiet: true });
    } catch { /* loaders surface the error */ }
  }, 2000);
}

function scheduleTrainingRefresh(training) {
  const status = liveValidation(training).status?.state;
  clearTimeout(catalogState.refreshPoll);
  catalogState.refreshPoll = null;
  if (!["pending", "refreshing"].includes(status)) return;
  catalogState.refreshPoll = setTimeout(async () => {
    const route = parseRoute();
    if (route.view !== "training" || route.trainingId !== training.id) return;
    try {
      await loadTrainingDetail(training.id, { quiet: true });
    } catch {
      // loadTrainingDetail surfaces the error.
    }
  }, 2000);
}

function trainingTableRow(training) {
  const checkpoint = checkpointStatus(training.latest_checkpoint);
  const rollouts = rolloutAvailable(training);
  const live = liveRollouts(training);
  const attemptCount = live.attempts?.length ?? (training.launches || []).reduce(
    (total, launch) => total + (launch.rollouts || []).reduce((subtotal, rollout) => subtotal + (rollout.present_attempts || []).length, 0), 0
  );
  return `<tr data-training-id="${escapeHtml(training.id)}" tabindex="0" aria-label="Open ${escapeHtml(training.title)}">
    <td class="training-identity">
      <div class="classification-line"><span class="classification-badge ${escapeHtml(training.classification)}">${escapeHtml(classificationLabel(training.classification))}</span>${training.visibility === "archive" ? '<span class="archive-label">archived</span>' : ""}</div>
      <h3>${escapeHtml(training.title)}</h3>
      <div class="tag-row">${tagPills(training.tags)}</div>
      <code>${escapeHtml(training.id)}</code>
    </td>
    <td class="step-cell"><strong>${formatNumber(training.furthest_step)}</strong><small>furthest step</small></td>
    <td class="score-cell">${bestFamilyEvaluation(training, "browsecomp-plus")}</td>
    <td class="score-cell">${bestFamilyEvaluation(training, "browsecomp")}</td>
    <td class="score-cell">${bestLiveValidation(training)}</td>
    <td class="availability-cell">
      <span><i class="status-dot ${rollouts ? "ready" : "missing"}"></i>${rollouts ? `${formatNumber(attemptCount)} rollout attempt${attemptCount === 1 ? "" : "s"}` : "no indexed rollouts"}</span>
      <span><i class="status-dot ${checkpoint.available ? "ready" : "missing"}"></i>${escapeHtml(checkpoint.label)}</span>
    </td>
    <td class="row-arrow" aria-hidden="true">→</td>
  </tr>`;
}

function renderTrainingTable() {
  const trainings = filteredTrainings();
  elements["catalog-count"].textContent = `${formatNumber(trainings.length)} of ${formatNumber(catalogState.trainings.length)} registered experiments`;
  elements["catalog-empty"].hidden = trainings.length !== 0;
  elements["training-table"].hidden = trainings.length === 0;
  elements["training-table"].innerHTML = trainings.length ? `<table class="training-table">
    <thead><tr><th>Experiment</th><th>Progress</th><th>Best TB2.1</th><th>Best TBlite</th><th>In-training validation</th><th>Artifacts</th><th><span class="sr-only">Open</span></th></tr></thead>
    <tbody>${trainings.map(trainingTableRow).join("")}</tbody>
  </table>` : "";
  elements["training-table"].querySelectorAll("tr[data-training-id]").forEach((row) => {
    const open = () => { window.location.hash = `#/trainings/${encodeURIComponent(row.dataset.trainingId)}`; };
    row.addEventListener("click", open);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); open(); }
    });
  });
}

async function loadCatalog({ force = false, quiet = false } = {}) {
  if (catalogState.loading) return;
  if (!force && catalogState.trainings.length) {
    renderCatalogSummary();
    renderTrainingTable();
    return;
  }
  catalogState.loading = true;
  if (!quiet) {
    elements["training-table"].innerHTML = '<div class="catalog-loading"><i></i><span>Reading the training registry…</span></div>';
  }
  try {
    const payload = await api("/api/trainings");
    catalogState.trainings = payload.trainings || [];
    catalogState.summary = payload.summary || {};
    catalogState.refreshedAt = payload.refreshed_at || new Date().toISOString();
    renderCatalogSummary();
    renderTrainingTable();
    updateRefreshProgress();
  } catch (error) {
    if (!quiet) {
      elements["training-table"].innerHTML = `<div class="empty-state"><h3>Registry could not be loaded</h3><p>${escapeHtml(error.message)}</p></div>`;
    }
    showToast(error.message);
  } finally {
    catalogState.loading = false;
  }
}

function artifactSection(training) {
  const bestCheckpoint = training.best_evaluation?.checkpoint;
  return `<section class="detail-card artifact-card">
    <div class="detail-card-heading"><div><p class="eyebrow">Retained state</p><h2>Checkpoints</h2></div></div>
    ${pathRow(
      `Best full-eval checkpoint · step ${bestCheckpoint?.step ?? "—"}`,
      bestCheckpoint?.path,
      { exists: bestCheckpoint?.complete ?? bestCheckpoint?.exists ?? false, url: bestCheckpoint?.path_url || bestCheckpoint?.url || bestCheckpoint?.link },
    ) || '<p class="metric-empty">No full-evaluation checkpoint recorded.</p>'}
    ${pathRow(
      `Latest checkpoint · step ${training.latest_checkpoint?.step ?? "—"}`,
      training.latest_checkpoint?.path,
      { exists: training.latest_checkpoint?.complete ?? training.latest_checkpoint?.exists ?? false, url: training.latest_checkpoint?.path_url || training.latest_checkpoint?.url || training.latest_checkpoint?.link },
    ) || '<p class="metric-empty">No latest checkpoint recorded.</p>'}
    ${(training.checkpoints || []).length ? `<div class="checkpoint-inventory"><p class="eyebrow">All retained checkpoints · ${training.checkpoints.length}</p><div class="checkpoint-chips">${training.checkpoints.map((checkpoint) => `<span class="checkpoint-chip ${checkpoint.complete ? "ready" : "missing"}" title="${escapeHtml(checkpoint.path)}${checkpoint.complete ? "" : " (missing or incomplete)"}">${formatNumber(checkpoint.step)}</span>`).join("")}</div></div>` : ""}
  </section>`;
}

function launchCard(launch) {
  const repository = String(launch.git_repository || "allenai/open-instruct")
    .replace(/^https:\/\/github\.com\//, "")
    .replace(/\.git$/, "")
    .replace(/\/$/, "");
  const gitLabel = launch.git_commit
    ? `${repository} @${String(launch.git_commit).slice(0, 8)}`
    : "Git commit";
  const links = [
    externalLink(launch.beaker_url, "Beaker"),
    externalLink(launch.wandb?.url, "W&B"),
    externalLink(launch.image_url, "Image"),
    externalLink(launch.git_url, gitLabel),
  ].filter(Boolean).join("");
  const scriptPath = launch.script_path || launch.script || launch.historical_script;
  const scriptLink = launch.script_url || launch.path_url || launch.script_link;
  const state = launch.checkpoint_state;
  const stateLabel = state
    ? `Checkpoint state · ${state.latest || "no latest"}${(state.resumable_steps || []).length ? ` · ${state.resumable_steps.length} resumable` : ""}`
    : "";
  const stateRow = state ? pathRow(stateLabel, state.path, { exists: state.exists }) : "";
  const rolloutRows = (launch.rollouts || []).map((rollout, index) => pathRow(
    `Rollouts${launch.rollouts.length > 1 ? ` ${index + 1}` : ""} · ${(rollout.present_attempts || []).length}/${(rollout.attempts || []).length} attempts present`,
    rollout.path,
    { exists: rollout.exists, url: rollout.path_url || rollout.url || rollout.link },
  )).join("");
  return `<article class="launch-card">
    <header><div><span class="relation-badge">${escapeHtml(launch.relation || "launch")}</span><h3>${escapeHtml(launch.id || "Launch")}</h3></div><div class="launch-links">${links}</div></header>
    ${pathRow(launch.historical_script && !launch.script ? "Historical launch script" : "Launch script", scriptPath, {
      exists: launch.script_exists,
      url: scriptLink,
    })}
    ${rolloutRows || '<p class="metric-empty">No persistent rollout path recorded.</p>'}
    ${stateRow}
    ${launch.note ? `<p class="launch-note">${escapeHtml(launch.note)}</p>` : ""}
  </article>`;
}

function chartValue(value, unit) {
  if (!Number.isFinite(value)) return "—";
  if (unit === "rate") return formatPercent(value);
  if (unit === "tokens") return formatTokens(value);
  return formatNumber(value, value < 10 ? 3 : 1);
}

function niceStep(span, targetTicks = 5, integer = false) {
  if (!Number.isFinite(span) || span <= 0) return integer ? 1 : 0.1;
  const rough = span / Math.max(1, targetTicks - 1);
  const power = 10 ** Math.floor(Math.log10(rough));
  const scaled = rough / power;
  const factor = integer
    ? scaled <= 1.5 ? 1 : scaled <= 3 ? 2 : scaled <= 7 ? 5 : 10
    : scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 2.5 ? 2.5 : scaled <= 5 ? 5 : 10;
  const step = factor * power;
  return integer ? Math.max(1, Math.ceil(step)) : step;
}

function niceYScale(values, unit, domain = null) {
  let rawMin = domain?.[0] ?? Math.min(...values);
  let rawMax = domain?.[1] ?? Math.max(...values);
  if (rawMin === rawMax) {
    const spread = Math.max(Math.abs(rawMin) * 0.1, unit === "rate" ? 0.025 : 0.01);
    rawMin -= spread;
    rawMax += spread;
  } else if (!domain) {
    const padding = (rawMax - rawMin) * 0.08;
    rawMin -= padding;
    rawMax += padding;
  }
  if (unit === "rate") {
    rawMin = Math.max(0, rawMin);
    rawMax = Math.min(1, rawMax);
  }
  const step = niceStep(rawMax - rawMin, 5);
  let min = Math.floor(rawMin / step) * step;
  let max = Math.ceil(rawMax / step) * step;
  if (unit === "rate") {
    min = Math.max(0, min);
    max = Math.min(1, max);
  }
  if (min === max) max = min + step;
  const ticks = [];
  for (let value = min, guard = 0; value <= max + step * 0.01 && guard < 12; value += step, guard += 1) {
    ticks.push(Number(value.toPrecision(12)));
  }
  return { min, max, ticks, step };
}

function niceXTicks(min, max) {
  if (min === max) return [min];
  const step = niceStep(max - min, 5, true);
  const ticks = [];
  const first = Math.ceil(min / step) * step;
  for (let value = first; value <= max; value += step) ticks.push(value);
  if (!ticks.includes(min)) ticks.unshift(min);
  if (!ticks.includes(max)) ticks.push(max);
  return ticks.filter((value, index, all) => index === 0 || value !== all[index - 1]);
}

function axisValue(value, unit, step) {
  if (unit === "rate") {
    const percentageStep = Math.abs(step * 100);
    const decimals = percentageStep < 1 || !Number.isInteger(percentageStep) ? 1 : 0;
    return `${(value * 100).toFixed(decimals)}%`;
  }
  if (unit === "tokens") return formatTokens(value);
  const decimals = Math.abs(step) < 0.1 ? 2 : Math.abs(step) < 1 ? 1 : 0;
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: decimals });
}

function chartMarkup({ title, subtitle, points = [], color = "#2563eb", unit = "count", domain = null }) {
  const clean = points
    .map((point) => ({ step: Number(point.optimizer_step), value: Number(point.value ?? point.score) }))
    .filter((point) => Number.isFinite(point.step) && Number.isFinite(point.value))
    .sort((left, right) => left.step - right.step);
  if (!clean.length) {
    return `<article class="metric-chart metric-chart-empty">
      <div class="metric-chart-heading"><div><h3>${escapeHtml(title)}</h3><p>${escapeHtml(subtitle)}</p></div><strong>Not logged</strong></div>
      <div class="chart-empty">No scalar history is available for this run.</div>
    </article>`;
  }
  const width = 520;
  const height = 220;
  const pad = { left: 58, right: 18, top: 14, bottom: 42 };
  const xMin = Math.min(...clean.map((point) => point.step));
  const xMax = Math.max(...clean.map((point) => point.step));
  const yScale = niceYScale(clean.map((point) => point.value), unit, domain);
  const { min: yMin, max: yMax } = yScale;
  const x = (step) => pad.left + ((step - xMin) / Math.max(1, xMax - xMin)) * (width - pad.left - pad.right);
  const y = (value) => pad.top + (1 - (value - yMin) / (yMax - yMin)) * (height - pad.top - pad.bottom);
  const path = clean.map((point, index) => `${index ? "L" : "M"}${x(point.step).toFixed(1)},${y(point.value).toFixed(1)}`).join(" ");
  const latest = clean.at(-1);
  const yGrid = yScale.ticks.map((value) => {
    const gridY = y(value);
    return `<line x1="${pad.left}" y1="${gridY}" x2="${width - pad.right}" y2="${gridY}" />
      <text x="${pad.left - 9}" y="${gridY + 3}" text-anchor="end">${escapeHtml(axisValue(value, unit, yScale.step))}</text>`;
  }).join("");
  const xTicks = niceXTicks(xMin, xMax);
  const xGrid = xTicks.map((value) => `<line x1="${x(value)}" y1="${pad.top}" x2="${x(value)}" y2="${height - pad.bottom}" />
    <text x="${x(value)}" y="${height - pad.bottom + 18}" text-anchor="middle">${formatNumber(value)}</text>`).join("");
  const dots = clean.length <= 36
    ? clean.map((point) => `<circle cx="${x(point.step)}" cy="${y(point.value)}" r="3"><title>Step ${point.step}: ${chartValue(point.value, unit)}</title></circle>`).join("")
    : `<circle cx="${x(latest.step)}" cy="${y(latest.value)}" r="4"><title>Step ${latest.step}: ${chartValue(latest.value, unit)}</title></circle>`;
  const interaction = escapeHtml(JSON.stringify({ points: clean, unit, width, height, pad, xMin, xMax, yMin, yMax }));
  return `<article class="metric-chart" style="--chart-color:${color}">
    <div class="metric-chart-heading"><div><h3>${escapeHtml(title)}</h3><p title="${escapeHtml(subtitle)}">${escapeHtml(subtitle)}</p></div><strong>${chartValue(latest.value, unit)}</strong></div>
    <svg viewBox="0 0 ${width} ${height}" role="img" tabindex="0" data-chart="${interaction}" aria-label="${escapeHtml(title)} from optimizer step ${xMin} to ${xMax}. Hover or use arrow keys to inspect values.">
      <g class="chart-grid chart-y-grid">${yGrid}</g>
      <g class="chart-grid chart-x-grid">${xGrid}</g>
      <path class="chart-line" d="${path}" />
      <g class="chart-dots">${dots}</g>
      <g class="chart-hover" hidden><line class="chart-crosshair" y1="${pad.top}" y2="${height - pad.bottom}" /><circle class="chart-hover-dot" r="5" /></g>
      <rect class="chart-hitbox" x="${pad.left}" y="${pad.top}" width="${width - pad.left - pad.right}" height="${height - pad.top - pad.bottom}" />
      <text class="chart-axis-title" x="${(pad.left + width - pad.right) / 2}" y="${height - 5}" text-anchor="middle">Optimizer step</text>
    </svg>
    <div class="chart-tooltip" role="tooltip" hidden><b></b><span></span></div>
  </article>`;
}

function wireMetricCharts(host) {
  host.querySelectorAll("svg[data-chart]").forEach((svg) => {
    const chart = JSON.parse(svg.dataset.chart);
    const article = svg.closest(".metric-chart");
    const hover = svg.querySelector(".chart-hover");
    const crosshair = svg.querySelector(".chart-crosshair");
    const dot = svg.querySelector(".chart-hover-dot");
    const tooltip = article.querySelector(".chart-tooltip");
    let selectedIndex = chart.points.length - 1;
    const x = (step) => chart.pad.left + ((step - chart.xMin) / Math.max(1, chart.xMax - chart.xMin)) * (chart.width - chart.pad.left - chart.pad.right);
    const y = (value) => chart.pad.top + (1 - (value - chart.yMin) / (chart.yMax - chart.yMin)) * (chart.height - chart.pad.top - chart.pad.bottom);

    const showPoint = (index) => {
      selectedIndex = Math.max(0, Math.min(chart.points.length - 1, index));
      const point = chart.points[selectedIndex];
      const pointX = x(point.step);
      const pointY = y(point.value);
      crosshair.setAttribute("x1", pointX);
      crosshair.setAttribute("x2", pointX);
      dot.setAttribute("cx", pointX);
      dot.setAttribute("cy", pointY);
      hover.hidden = false;
      tooltip.querySelector("b").textContent = chartValue(point.value, chart.unit);
      tooltip.querySelector("span").textContent = `Optimizer step ${formatNumber(point.step)}`;
      tooltip.hidden = false;
      const svgWidth = svg.getBoundingClientRect().width;
      const left = (pointX / chart.width) * svgWidth + svg.offsetLeft;
      const top = (pointY / chart.height) * svg.getBoundingClientRect().height + svg.offsetTop;
      tooltip.style.left = `${Math.max(72, Math.min(article.clientWidth - 72, left))}px`;
      tooltip.style.top = `${Math.max(54, top)}px`;
    };
    const hidePoint = () => {
      hover.hidden = true;
      tooltip.hidden = true;
    };
    svg.addEventListener("pointermove", (event) => {
      const bounds = svg.getBoundingClientRect();
      const pointerX = ((event.clientX - bounds.left) / bounds.width) * chart.width;
      const index = chart.points.reduce((best, point, candidate) =>
        Math.abs(x(point.step) - pointerX) < Math.abs(x(chart.points[best].step) - pointerX) ? candidate : best, 0);
      showPoint(index);
    });
    svg.addEventListener("pointerleave", hidePoint);
    svg.addEventListener("focus", () => showPoint(selectedIndex));
    svg.addEventListener("blur", hidePoint);
    svg.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      if (event.key === "Home") selectedIndex = 0;
      else if (event.key === "End") selectedIndex = chart.points.length - 1;
      else selectedIndex += event.key === "ArrowRight" ? 1 : -1;
      showPoint(selectedIndex);
    });
  });
}

function renderTrainingCharts(training, payload = null) {
  const host = document.getElementById("training-charts");
  if (!host || catalogState.trainingId !== training.id) return;
  const validation = liveValidation(training);
  const series = payload?.series || {};
  const validationPoints = (validation.evaluations || []).map((item) => ({
    optimizer_step: validationStep(item),
    value: validationScore(item),
  }));
  const capped = series.token_capped?.points || [];
  const latestCapped = capped.at(-1)?.value;
  const logprobMetric = series.logprob?.metric;
  const logprobTitle = logprobMetric === "policy/entropy_avg" ? "Policy entropy" : "Logprob drift";
  const logprobSubtitle = logprobMetric || "debug/vllm_vs_local_logprob_diff_mean";
  const preMaskGroupPoints = series.group_pass_rate_all?.points || [];
  const postMaskGroupPoints = series.group_pass_rate_post_mask?.points || [];
  const formatCharts = [
    {
      key: "format_incomplete_pre",
      title: "Incomplete rollouts · pre-filter",
      metric: "format/incomplete_pre_filtering",
      color: "#dc2626",
    },
    {
      key: "format_terminal_pre",
      title: "Terminal-format errors · pre-filter",
      metric: "format/terminal_format_pre_filtering",
      color: "#ea580c",
    },
    {
      key: "format_trajectory_pre",
      title: "Trajectory-format errors · pre-filter",
      metric: "format/trajectory_format_pre_filtering",
      color: "#9333ea",
    },
    {
      key: "format_incomplete_post",
      title: "Incomplete rollouts · post-filter",
      metric: "format/incomplete_post_filtering",
      color: "#b91c1c",
    },
    {
      key: "format_terminal_post",
      title: "Terminal-format errors · post-filter",
      metric: "format/terminal_format_post_filtering",
      color: "#c2410c",
    },
    {
      key: "format_trajectory_post",
      title: "Trajectory-format errors · post-filter",
      metric: "format/trajectory_format_post_filtering",
      color: "#7e22ce",
    },
  ];
  const hasFormatMetrics = formatCharts.some(({ key }) => (series[key]?.points || []).length > 0);
  const groupMaskChanged =
    postMaskGroupPoints.length > 0 &&
    (preMaskGroupPoints.length !== postMaskGroupPoints.length ||
      postMaskGroupPoints.some(
        (point, index) =>
          point.optimizer_step !== preMaskGroupPoints[index]?.optimizer_step ||
          Math.abs(point.value - preMaskGroupPoints[index]?.value) > 1e-12,
      ));
  host.innerHTML = [
    chartMarkup({
      title: "Validation score",
      subtitle: validation.latest?.metric || training.wandb?.validation_metric || "held-out evaluation",
      points: validationPoints,
      color: "#0f766e",
      unit: "rate",
    }),
    chartMarkup({
      title: "Training reward",
      subtitle: series.reward?.metric || "scores",
      points: series.reward?.points,
      color: "#2563eb",
      unit: "rate",
      domain: [0, 1],
    }),
    chartMarkup({
      title: "Mean group pass rate · all sampled (pre-mask)",
      subtitle: series.group_pass_rate_all?.metric || "val/avg_group_performance_pre_filter",
      points: series.group_pass_rate_all?.points,
      color: "#0891b2",
      unit: "rate",
      domain: [0, 1],
    }),
    ...(groupMaskChanged
      ? [
          chartMarkup({
            title: "Mean group pass rate · all sampled (post-mask)",
            subtitle: series.group_pass_rate_post_mask?.metric || "val/avg_group_performance_post_filter",
            points: postMaskGroupPoints,
            color: "#7c3aed",
            unit: "rate",
          }),
        ]
      : []),
    chartMarkup({
      title: "Rejected group rate",
      subtitle: "rejected / (accepted + rejected)",
      points: series.rejected_group_rate?.points,
      color: "#dc2626",
      unit: "rate",
      domain: [0, 1],
    }),
    chartMarkup({
      title: "Rejected groups: all 1s",
      subtitle: "all-1 rejected groups / rejected groups",
      points: series.rejected_all_one_rate?.points,
      color: "#059669",
      unit: "rate",
      domain: [0, 1],
    }),
    chartMarkup({
      title: "Rejected groups: all 0s",
      subtitle: "all-0 rejected groups / rejected groups",
      points: series.rejected_all_zero_rate?.points,
      color: "#d97706",
      unit: "rate",
      domain: [0, 1],
    }),
    ...(hasFormatMetrics
      ? formatCharts.map(({ key, title, metric, color }) =>
          chartMarkup({
            title,
            subtitle: series[key]?.metric || metric,
            points: series[key]?.points,
            color,
            unit: "rate",
          }),
        )
      : []),
    chartMarkup({
      title: "Average trajectory length",
      subtitle: `${series.length?.metric || "val/sequence_lengths"}${Number.isFinite(latestCapped) ? ` · latest capped ${formatPercent(latestCapped)}` : ""}`,
      points: series.length?.points,
      color: "#7c3aed",
      unit: "tokens",
    }),
    chartMarkup({
      title: "Average terminal turn length",
      subtitle: series.terminal_length?.metric || "val/terminal_turn_lengths",
      points: series.terminal_length?.points,
      color: "#0f766e",
      unit: "tokens",
    }),
    chartMarkup({
      title: "Tool calls per rollout",
      subtitle: series.tool_calls?.metric || "tools/aggregate/avg_calls_per_rollout",
      points: series.tool_calls?.points,
      color: "#b45309",
      unit: "count",
    }),
    chartMarkup({
      title: "Primary tool failure rate",
      subtitle: series.search_failure_rate?.metric || "tools/search/failure_rate",
      points: series.search_failure_rate?.points,
      color: "#e11d48",
      unit: "rate",
    }),
    chartMarkup({
      title: "Aggregate tool failure rate",
      subtitle: series.visit_failure_rate?.metric || "tools/visit/failure_rate",
      points: series.visit_failure_rate?.points,
      color: "#c026d3",
      unit: "rate",
    }),
    chartMarkup({
      title: logprobTitle,
      subtitle: logprobSubtitle,
      points: series.logprob?.points,
      color: "#be123c",
      unit: "count",
    }),
  ].join("");
  wireMetricCharts(host);
}

function renderTrainingDetail(training, metrics = null) {
  catalogState.trainingId = training.id;
  elements["training-breadcrumb"].textContent = training.title;
  const validation = liveValidation(training);
  const latestValidation = validation.latest || (validation.evaluations || []).at(-1);
  const rollouts = liveRollouts(training);
  const hasRollouts = rolloutAvailable(training);
  const checkpoint = checkpointStatus(training.latest_checkpoint);
  const validationStatus = typeof validation.status === "string" ? validation.status : validation.status?.state;
  const validationNote = ["error", "stale"].includes(validationStatus)
    ? `Refresh error${validation.status?.error ? `: ${validation.status.error}` : ""}`
    : validation.status?.fetched_at ? `Checked ${formatDate(validation.status.fetched_at)}` : "W&B validation history";
  const latestLaunch = (training.launches || []).at(-1);
  const activity = trainingActivity(training);
  elements["training-detail"].innerHTML = `
    <header class="training-detail-header">
      <div>
        <div class="classification-line"><span class="classification-badge ${escapeHtml(training.classification)}">${escapeHtml(classificationLabel(training.classification))}</span>${training.visibility === "archive" ? '<span class="archive-label">archived</span>' : ""}</div>
        <h1>${escapeHtml(training.title)}</h1>
        <div class="tag-row">${tagPills(training.tags)}</div>
        <code class="training-id">${escapeHtml(training.id)}</code>
      </div>
      <div class="detail-actions">
        ${training.wandb?.url ? externalLink(training.wandb.url, "Open W&B", "primary-link") : ""}
        ${latestLaunch?.beaker_url ? externalLink(latestLaunch.beaker_url, "Beaker", "primary-link secondary-link") : ""}
      </div>
    </header>
    ${training.note ? `<p class="training-note">${escapeHtml(training.note)}</p>` : ""}
    <section class="detail-metrics" aria-label="Training summary">
      <article><span>Furthest step</span><strong>${formatNumber(training.furthest_step)}</strong><small>optimizer steps</small></article>
      <article><span>Best full evaluation</span>${bestFullEval(training)}</article>
      <article><span>Best in-training validation</span>${bestLiveValidation(training)}<small>${escapeHtml(validationNote)}</small></article>
      <article><span>Latest validation</span><strong>${formatPercent(validationScore(latestValidation))}</strong><small>${validationStep(latestValidation) === null ? "no scored step" : `step ${formatNumber(validationStep(latestValidation))}`}</small></article>
      <article><span>Latest checkpoint</span><strong>${escapeHtml(checkpoint.label)}</strong><small>${checkpoint.available ? "complete on disk" : "not available"}</small></article>
      <article><span>Rollout artifacts</span><strong>${hasRollouts ? "Available" : "Unavailable"}</strong><small>${formatNumber(rollouts.attempts?.length ?? 0)} indexed attempts</small></article>
    </section>
    ${fullEvaluationSection(training)}
    <section class="metric-section" aria-labelledby="metric-section-title">
      <div class="section-title-row">
        <div><p class="eyebrow">Training dynamics</p><h2 id="metric-section-title">Run metrics</h2></div>
        <span class="muted">${activity ? `Artifacts updated ${formatDate(activity)}` : "Historical W&B scalars"}</span>
      </div>
      <div id="training-charts" class="metric-chart-grid"><div class="chart-loading">Loading W&B history…</div></div>
    </section>
    <details class="run-details">
      <summary><span><b>Run details</b><small>Launch history, checkpoints, scripts, images, and artifact paths</small></span><i aria-hidden="true">⌄</i></summary>
      <div class="training-detail-grid">
        <section class="detail-card launch-section">
          <div class="detail-card-heading"><div><p class="eyebrow">Provenance</p><h2>Launch history</h2></div><span>${formatNumber((training.launches || []).length)} launch${(training.launches || []).length === 1 ? "" : "es"}</span></div>
          <div class="launch-list">${(training.launches || []).map(launchCard).join("") || '<p class="metric-empty">No launch record.</p>'}</div>
        </section>
        ${artifactSection(training)}
      </div>
    </details>`;
  renderTrainingCharts(training, metrics);
  wireCopyButtons(elements["training-detail"]);
}

async function loadTrainingDetail(trainingId, { quiet = false } = {}) {
  const requestVersion = ++catalogState.detailRequestVersion;
  if (!quiet) elements["training-detail"].innerHTML = '<div class="catalog-loading"><i></i><span>Loading experiment provenance…</span></div>';
  try {
    const encodedId = encodeURIComponent(trainingId);
    const cachedMetrics = catalogState.metricsByTraining[trainingId];
    const metrics = cachedMetrics || await api(`/api/trainings/${encodedId}/metrics`).catch(() => null);
    if (requestVersion !== catalogState.detailRequestVersion) return null;
    const payload = await api(`/api/trainings/${encodedId}`);
    if (requestVersion !== catalogState.detailRequestVersion) return null;
    const training = payload.training || payload;
    if (metrics) catalogState.metricsByTraining[trainingId] = metrics;
    renderTrainingDetail(training, metrics);
    scheduleTrainingRefresh(training);
    return training;
  } catch (error) {
    if (requestVersion !== catalogState.detailRequestVersion) return null;
    elements["training-detail"].innerHTML = `<div class="empty-state"><h3>Experiment could not be loaded</h3><p>${escapeHtml(error.message)}</p></div>`;
    showToast(error.message);
    return null;
  }
}

async function copyText(value) {
  try {
    await navigator.clipboard.writeText(value);
  } catch {
    const input = document.createElement("textarea");
    input.value = value;
    input.style.position = "fixed";
    input.style.opacity = "0";
    document.body.appendChild(input);
    input.select();
    document.execCommand("copy");
    input.remove();
  }
}

function wireCopyButtons(host = document) {
  host.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      await copyText(button.dataset.copy);
      const previous = button.textContent;
      button.textContent = "Copied";
      setTimeout(() => { button.textContent = previous; }, 1200);
    });
  });
}

function absoluteViewerUrl(hash) {
  return `${window.location.origin}${window.location.pathname}${hash}`;
}

function trainingRolloutHash(trainingId, detail) {
  return `#/trainings/${encodeURIComponent(trainingId)}/rollouts/${encodeURIComponent(detail.source)}/${encodeURIComponent(detail.step)}/${encodeURIComponent(detail.id)}`;
}

function evaluationRolloutHash(trainingId, evaluationId, queryId, responseIndex) {
  return `#/trainings/${encodeURIComponent(trainingId)}/evaluations/${encodeURIComponent(evaluationId)}/records/${encodeURIComponent(queryId)}/${encodeURIComponent(responseIndex)}`;
}

function trainingRolloutLocator(trainingId, detail) {
  const hash = trainingRolloutHash(trainingId, detail);
  return [
    "Training rollout locator",
    `training_id: ${trainingId}`,
    `artifact_step: ${detail.step}`,
    `optimizer_step: ${Number(detail.step) + 1}`,
    `source: ${detail.source}`,
    `prompt_idx: ${detail.prompt_idx}`,
    `sample_idx: ${detail.sample_idx}`,
    `record_id: ${detail.id}`,
    `reference_answer: ${detail.ground_truth || ""}`,
    `url: ${absoluteViewerUrl(hash)}`,
  ].join("\n");
}

function evaluationRolloutLocator(payload) {
  const hash = evaluationRolloutHash(
    payload.training_id,
    payload.evaluation.id,
    payload.record.query_id,
    payload.selected_response_index,
  );
  return [
    "Evaluation rollout locator",
    `training_id: ${payload.training_id}`,
    `evaluation_id: ${payload.evaluation.id}`,
    `benchmark: ${payload.evaluation.benchmark}`,
    `checkpoint_step: ${payload.evaluation.step}`,
    `query_id: ${payload.record.query_id}`,
    `response_index: ${payload.selected_response_index}`,
    `judged_response_index: ${payload.judged_response_index}`,
    `url: ${absoluteViewerUrl(hash)}`,
  ].join("\n");
}

function wireLocatorButton(button, locator) {
  if (!button) return;
  button.hidden = false;
  button.disabled = false;
  button.textContent = "Copy rollout locator";
  button.onclick = async () => {
    await copyText(locator);
    button.textContent = "Copied locator";
    setTimeout(() => {
      if (button.isConnected) button.textContent = "Copy rollout locator";
    }, 1200);
  };
}

function evaluationOutcomeBadge(record, judgeApplies = true) {
  if (!judgeApplies) return '<span class="evaluation-outcome neutral">Not judged</span>';
  if (record.outcome === "incomplete" || record.completed === false) return '<span class="evaluation-outcome incomplete">Incomplete</span>';
  if (record.outcome === "judged_correct" || record.correct === true) return '<span class="evaluation-outcome correct">Judged correct</span>';
  if (record.outcome === "judged_incorrect" || record.correct === false) return '<span class="evaluation-outcome incorrect">Judged incorrect</span>';
  return '<span class="evaluation-outcome neutral">Missing verdict</span>';
}

function evaluationToolSummary(counts) {
  const entries = Object.entries(counts || {}).filter(([, value]) => Number(value) > 0);
  return entries.length ? entries.map(([name, value]) => `${formatNumber(value)} ${name}`).join(" · ") : "No tool calls";
}

const EVALUATION_MATCH_CATEGORIES = [
  ["reasoning", "Model reasoning/text"],
  ["tool_call", "Tool calls"],
  ["tool_result", "Tool responses"],
  ["final_output", "Final output"],
];

function evaluationMatchChips(result) {
  const counts = result?.counts || {};
  return EVALUATION_MATCH_CATEGORIES.map(([key, label]) => `
    <span class="evaluation-match-chip"><strong>${formatNumber(counts[key] || 0)}</strong>${escapeHtml(label)}</span>`).join("");
}

function evaluationFinderPanel(payload, counts, assistantTextNote) {
  const reference = payload.reference_matches || { term: payload.record?.reference_answer || "", total: 0, counts: {} };
  const hasReference = Boolean(String(reference.term || "").trim());
  const documents = payload.document_relevance || {};
  const documentPanel = documents.available ? `<div class="evaluation-document-find" aria-label="BrowseComp-Plus source navigation">
    <div class="evaluation-document-nav evidence">
      <span class="document-nav-symbol" aria-hidden="true">●</span>
      <div><strong>Evidence results</strong><small>${formatNumber(documents.matched_evidence_url_count || 0)} of ${formatNumber(documents.evidence_url_count || 0)} URLs found</small></div>
      <div class="evaluation-find-nav">
        <button type="button" class="icon-button" data-document-nav="evidence-prev" aria-label="Previous tool result containing evidence"${documents.evidence_result_count ? "" : " disabled"}>↑</button>
        <button type="button" class="icon-button" data-document-nav="evidence-next" aria-label="Next tool result containing evidence"${documents.evidence_result_count ? "" : " disabled"}>↓</button>
        <output data-document-count="evidence" aria-live="polite">${formatNumber(documents.evidence_result_count || 0)} result${documents.evidence_result_count === 1 ? "" : "s"}</output>
      </div>
    </div>
    <div class="evaluation-document-nav gold">
      <span class="document-nav-symbol" aria-hidden="true">★</span>
      <div><strong>Gold results</strong><small>${formatNumber(documents.matched_positive_url_count || 0)} of ${formatNumber(documents.positive_url_count || 0)} URLs found</small></div>
      <div class="evaluation-find-nav">
        <button type="button" class="icon-button" data-document-nav="gold-prev" aria-label="Previous tool result containing a gold document"${documents.positive_result_count ? "" : " disabled"}>↑</button>
        <button type="button" class="icon-button" data-document-nav="gold-next" aria-label="Next tool result containing a gold document"${documents.positive_result_count ? "" : " disabled"}>↓</button>
        <output data-document-count="gold" aria-live="polite">${formatNumber(documents.positive_result_count || 0)} result${documents.positive_result_count === 1 ? "" : "s"}</output>
      </div>
    </div>
  </div>` : "";
  return `<section class="evaluation-find-panel${documents.available ? "" : " no-documents"}" aria-label="Find within trajectory">
    <div class="evaluation-reference-find">
      <div class="evaluation-find-title">
        <span>Reference answer in trajectory</span>
        <button class="evaluation-reference-term" type="button" data-activate-reference${hasReference ? "" : " disabled"} title="Restore reference-answer highlights">${escapeHtml(reference.term || "No reference answer")}</button>
      </div>
      <div class="evaluation-match-counts">${evaluationMatchChips(reference)}</div>
      <div class="evaluation-find-nav">
        <button type="button" class="icon-button" data-match-nav="reference-prev" aria-label="Previous reference-answer match"${reference.total ? "" : " disabled"}>↑</button>
        <button type="button" class="icon-button" data-match-nav="reference-next" aria-label="Next reference-answer match"${reference.total ? "" : " disabled"}>↓</button>
        <output data-match-count="reference" aria-live="polite">${formatNumber(reference.total || 0)} match${reference.total === 1 ? "" : "es"}</output>
      </div>
    </div>
    <div class="evaluation-trajectory-find">
      <div class="evaluation-search-heading"><strong>Search trajectory</strong><span>Reasoning, tools, and final output</span></div>
      <label class="evaluation-search-field">
        <span class="sr-only">Search this trajectory</span>
        <span class="evaluation-search-icon" aria-hidden="true"></span>
        <input type="search" data-trajectory-search maxlength="256" placeholder="Find text in this trajectory…" autocomplete="off" spellcheck="false" />
      </label>
      <div class="evaluation-search-footer">
        <output data-match-count="search" aria-live="polite">Enter text to search</output>
        <div class="evaluation-find-nav">
          <button type="button" class="icon-button" data-match-nav="search-prev" aria-label="Previous search match" disabled>↑</button>
          <button type="button" class="icon-button" data-match-nav="search-next" aria-label="Next search match" disabled>↓</button>
        </div>
      </div>
    </div>
    ${documentPanel}
    <div class="evaluation-sticky-turns">
      <div><div class="turns-summary prominent">${counts || "No structured turns"}</div>${assistantTextNote}</div>
      <div class="turns-actions"><button class="link-button" data-turns-action="expand">Expand all</button><button class="link-button" data-turns-action="collapse">Collapse all</button><button class="link-button" data-turns-action="tools">Hide tool results</button></div>
    </div>
  </section>`;
}

function highlightedMatchText(text, matches, currentIndex, source, rangeStart = 0, rangeEnd = text.length) {
  let cursor = rangeStart;
  const chunks = [];
  [...matches].sort((left, right) => left.start - right.start).forEach((match) => {
    const start = Math.max(cursor, Math.min(rangeEnd, Number(match.start)));
    const end = Math.max(start, Math.min(rangeEnd, Number(match.end)));
    if (end <= start) return;
    chunks.push(escapeHtml(text.slice(cursor, start)));
    chunks.push(`<mark class="trajectory-match ${source}${match.match_index === currentIndex ? " current" : ""}" data-match-index="${match.match_index}">${escapeHtml(text.slice(start, end))}</mark>`);
    cursor = end;
  });
  chunks.push(escapeHtml(text.slice(cursor, rangeEnd)));
  return chunks.join("");
}

function highlightedSegmentText(segment, matches = [], currentIndex = -1, source = "reference") {
  const text = String(segment.content || "");
  const regions = (segment.document_regions || [])
    .filter((region) => region.in_preview && Number(region.end) > Number(region.start))
    .sort((left, right) => Number(left.start) - Number(right.start));
  if (!regions.length) return highlightedMatchText(text, matches, currentIndex, source);
  const chunks = [];
  let cursor = 0;
  regions.forEach((region) => {
    const start = Math.max(cursor, Math.min(text.length, Number(region.start)));
    const end = Math.max(start, Math.min(text.length, Number(region.end)));
    if (end <= start) return;
    chunks.push(highlightedMatchText(text, matches, currentIndex, source, cursor, start));
    const label = region.kind === "gold" ? "Gold document" : "Evidence document";
    chunks.push(`<span class="trajectory-document ${escapeHtml(region.kind)}" aria-label="${label}">${highlightedMatchText(text, matches, currentIndex, source, start, end)}</span>`);
    cursor = end;
  });
  chunks.push(highlightedMatchText(text, matches, currentIndex, source, cursor, text.length));
  return chunks.join("");
}

function renderMatchContext(match, currentIndex, source) {
  const excerpt = String(match.excerpt || "");
  const start = Number(match.excerpt_match_start || 0);
  const end = Number(match.excerpt_match_end || start);
  return `<div class="turn-match-context"><span>Match beyond the displayed preview</span><pre>${escapeHtml(excerpt.slice(0, start))}<mark class="trajectory-match ${source}${match.match_index === currentIndex ? " current" : ""}" data-match-index="${match.match_index}">${escapeHtml(excerpt.slice(start, end))}</mark>${escapeHtml(excerpt.slice(end))}</pre></div>`;
}

function wireEvaluationFinder(host, payload, { searchMatches = null } = {}) {
  const reference = payload.reference_matches || { term: "", total: 0, counts: {}, matches: [] };
  let activeResult = reference;
  let activeSource = "reference";
  let currentIndex = -1;
  let searchResult = null;
  let searchVersion = 0;
  let searchController = null;
  let debounce = null;
  let paintedResult = null;
  const input = host.querySelector("[data-trajectory-search]");
  const referenceCount = host.querySelector('[data-match-count="reference"]');
  const searchCount = host.querySelector('[data-match-count="search"]');

  function normalizedResult(result) {
    const matches = (result?.matches || []).map((match, index) => ({ ...match, match_index: index }));
    return {
      ...(result || {}),
      total: Number(result?.total ?? matches.length),
      returned: Number(result?.returned ?? matches.length),
      truncated: Boolean(result?.truncated ?? false),
      matches,
    };
  }

  const normalizedReference = normalizedResult(reference);
  activeResult = normalizedReference;

  function updateNav(source, result, index) {
    const total = Number(result?.total || 0);
    const available = result?.matches?.length || 0;
    const output = source === "reference" ? referenceCount : searchCount;
    if (output) {
      if (!total) output.textContent = source === "search" && input?.value ? "No matches" : source === "search" ? "Enter text to search" : "0 matches";
      else if (index >= 0 && result?.truncated) output.textContent = `${formatNumber(index + 1)} / ${formatNumber(available)} · ${formatNumber(total)} total`;
      else if (index >= 0) output.textContent = `${formatNumber(index + 1)} / ${formatNumber(total)}`;
      else if (result?.truncated) output.textContent = `${formatNumber(total)} matches · first ${formatNumber(available)} navigable`;
      else output.textContent = `${formatNumber(total)} match${total === 1 ? "" : "es"}`;
    }
    host.querySelectorAll(`[data-match-nav^="${source}-"]`).forEach((button) => { button.disabled = available === 0; });
  }

  function paint(result, index) {
    const bySegment = new Map();
    (result?.matches || []).forEach((match) => {
      const key = Number(match.segment_index);
      if (!bySegment.has(key)) bySegment.set(key, []);
      bySegment.get(key).push(match);
    });
    (payload.segments || []).forEach((segment) => {
      const block = host.querySelector(`.turn-block[data-segment-index="${segment.index}"]`);
      if (!block) return;
      const content = block.querySelector(".turn-content");
      const matches = bySegment.get(Number(segment.index)) || [];
      const visible = matches.filter((match) => match.in_preview);
      if (content) content.innerHTML = highlightedSegmentText(segment, visible, index, activeSource);
      block.querySelector(".turn-match-contexts")?.remove();
      const hidden = matches.filter((match) => !match.in_preview);
      if (hidden.length) block.insertAdjacentHTML("beforeend", `<div class="turn-match-contexts">${hidden.map((match) => renderMatchContext(match, index, activeSource)).join("")}</div>`);
      block.classList.toggle("has-trajectory-match", matches.length > 0);
    });
    paintedResult = result;
  }

  function updateCurrent(index) {
    host.querySelectorAll(".trajectory-match.current").forEach((mark) => mark.classList.remove("current"));
    if (index < 0) return;
    host.querySelector(`.trajectory-match[data-match-index="${index}"]`)?.classList.add("current");
  }

  function activate(source, result, index = -1) {
    activeSource = source;
    activeResult = result;
    currentIndex = index;
    host.dataset.activeMatchSource = source;
    if (paintedResult !== result) paint(result, index);
    else updateCurrent(index);
    updateNav("reference", normalizedReference, source === "reference" ? index : -1);
    updateNav("search", searchResult, source === "search" ? index : -1);
  }

  function reveal(delta) {
    const available = activeResult?.matches?.length || 0;
    if (!available) return;
    currentIndex = currentIndex < 0 ? (delta > 0 ? 0 : available - 1) : (currentIndex + delta + available) % available;
    activate(activeSource, activeResult, currentIndex);
    const mark = host.querySelector(`.trajectory-match[data-match-index="${currentIndex}"]`);
    if (!mark) return;
    const block = mark.closest(".turn-block");
    if (block) {
      block.open = true;
      if (block.dataset.kind === "tool_result" && host.classList.contains("hide-tools")) {
        host.classList.remove("hide-tools");
        const toolsButton = host.querySelector('[data-turns-action="tools"]');
        if (toolsButton) {
          toolsButton.textContent = "Hide tool results";
          toolsButton.classList.remove("active");
        }
      }
    }
    mark.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  const documentTargets = {
    evidence: (payload.segments || []).filter((segment) => Number(segment.document_match_counts?.evidence || 0) > 0),
    gold: (payload.segments || []).filter((segment) => Number(segment.document_match_counts?.positive || 0) > 0),
  };
  const documentPositions = { evidence: -1, gold: -1 };

  function revealDocument(kind, delta) {
    const targets = documentTargets[kind] || [];
    if (!targets.length) return;
    const previous = documentPositions[kind];
    documentPositions[kind] = previous < 0 ? (delta > 0 ? 0 : targets.length - 1) : (previous + delta + targets.length) % targets.length;
    const target = targets[documentPositions[kind]];
    const block = host.querySelector(`.turn-block[data-segment-index="${target.index}"]`);
    if (!block) return;
    block.open = true;
    if (host.classList.contains("hide-tools")) {
      host.classList.remove("hide-tools");
      const toolsButton = host.querySelector('[data-turns-action="tools"]');
      if (toolsButton) {
        toolsButton.textContent = "Hide tool results";
        toolsButton.classList.remove("active");
      }
    }
    host.querySelectorAll(".turn-block.current-document-target").forEach((item) => item.classList.remove("current-document-target"));
    block.classList.add("current-document-target");
    const output = host.querySelector(`[data-document-count="${kind}"]`);
    if (output) output.textContent = `${formatNumber(documentPositions[kind] + 1)} / ${formatNumber(targets.length)}`;
    block.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  async function searchTrajectory(query) {
    const version = ++searchVersion;
    searchController?.abort();
    searchController = null;
    if (!query) {
      searchResult = null;
      activate("reference", normalizedReference);
      return;
    }
    if (searchCount) searchCount.textContent = "Searching…";
    const controller = new AbortController();
    searchController = controller;
    try {
      const result = searchMatches
        ? await searchMatches(query, controller.signal)
        : await api(`/api/trainings/${encodeURIComponent(evaluationState.trainingId)}/evaluations/${encodeURIComponent(evaluationState.evaluationId)}/records/${encodeURIComponent(payload.record.query_id)}/matches?${new URLSearchParams({ query, response_index: String(payload.selected_response_index) })}`, { signal: controller.signal });
      if (version !== searchVersion || !host.isConnected || input?.value.trim() !== query) return;
      searchResult = normalizedResult(result);
      activate("search", searchResult);
    } catch (error) {
      if (error.name === "AbortError" || version !== searchVersion || !host.isConnected) return;
      searchResult = null;
      activate("reference", normalizedReference);
      if (searchCount) searchCount.textContent = "Search failed";
      showToast(error.message);
    } finally {
      if (searchController === controller) searchController = null;
    }
  }

  host.querySelector('[data-match-nav="reference-prev"]')?.addEventListener("click", () => { if (activeSource !== "reference") activate("reference", normalizedReference); reveal(-1); });
  host.querySelector('[data-match-nav="reference-next"]')?.addEventListener("click", () => { if (activeSource !== "reference") activate("reference", normalizedReference); reveal(1); });
  host.querySelector("[data-activate-reference]")?.addEventListener("click", () => activate("reference", normalizedReference));
  host.querySelector('[data-match-nav="search-prev"]')?.addEventListener("click", () => { if (searchResult) activate("search", searchResult, activeSource === "search" ? currentIndex : -1); reveal(-1); });
  host.querySelector('[data-match-nav="search-next"]')?.addEventListener("click", () => { if (searchResult) activate("search", searchResult, activeSource === "search" ? currentIndex : -1); reveal(1); });
  host.querySelectorAll("[data-document-nav]").forEach((button) => {
    const [kind, direction] = button.dataset.documentNav.split("-");
    button.addEventListener("click", () => revealDocument(kind, direction === "prev" ? -1 : 1));
  });
  input?.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => searchTrajectory(input.value.trim()), 180);
  });
  input?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      if (searchResult) {
        if (activeSource !== "search") activate("search", searchResult);
        reveal(event.shiftKey ? -1 : 1);
      }
    } else if (event.key === "Escape") {
      input.value = "";
      searchTrajectory("");
    }
  });
  host._evaluationFinderCleanup = () => {
    clearTimeout(debounce);
    searchVersion += 1;
    searchController?.abort();
    searchController = null;
  };
  activate("reference", normalizedReference);
}

function renderEvaluationHeader(training, payload) {
  const evaluation = payload.evaluation;
  const summary = payload.summary || {};
  const counts = payload.counts || {};
  const judgedCorrect = Number(counts.judged_correct ?? counts.correct ?? 0);
  const judgedIncorrect = Number(counts.judged_incorrect ?? counts.incorrect ?? 0);
  const incomplete = Number(counts.incomplete ?? 0);
  const accounted = Number(counts.accounted ?? (judgedCorrect + judgedIncorrect + incomplete));
  const expected = Number(evaluation.total || summary.num_results || 0);
  const accountingDelta = Number(counts.accounting_delta ?? (expected - accounted));
  const averageCalls = Object.values(summary.avg_tool_call_counts || {}).reduce((sum, value) => sum + Number(value || 0), 0);
  elements["evaluation-training-link"].href = `#/trainings/${encodeURIComponent(training.id)}`;
  elements["evaluation-training-link"].textContent = training.title;
  elements["evaluation-breadcrumb"].textContent = `${evaluation.benchmark} · step ${evaluation.step}`;
  elements["evaluation-header"].innerHTML = `<div>
    <p class="eyebrow">External evaluation trajectory examiner</p>
    <h1>${escapeHtml(training.title)}</h1>
    <p>${escapeHtml(evaluation.benchmark)} · checkpoint step ${formatNumber(evaluation.step)}</p>
  </div><div class="evaluation-header-actions"><a class="primary-link secondary-link" href="#/trainings/${encodeURIComponent(training.id)}">Back to training</a></div>`;
  elements["evaluation-stats"].innerHTML = `
    <article><span>Accuracy</span><strong>${formatPercent(evaluation.score, 2)}</strong><small>${formatNumber(evaluation.correct)}/${formatNumber(evaluation.total)} correct</small></article>
    <article><span>Evaluated questions</span><strong>${formatNumber(summary.num_results ?? evaluation.total)}</strong><small>${formatPercent(summary.avg_complete_rate)} complete</small></article>
    <article><span>Average tool calls</span><strong>${formatNumber(averageCalls, 1)}</strong><small>${escapeHtml(evaluationToolSummary(summary.avg_tool_call_counts))}</small></article>
    <article><span>Responses per question</span><strong>1+</strong><small>Compact switcher appears only for multi-rollout records</small></article>
    <article class="wide"><span>Registered artifact</span><strong>${escapeHtml(evaluation.inference_artifact?.display_path || "Unavailable")}</strong><small>Judge result joined to its raw inference trajectory</small></article>
    ${accountingDelta !== 0 ? `<aside class="evaluation-accounting-warning"><strong>Result accounting does not match the benchmark total.</strong><span>${formatNumber(judgedCorrect)} judged correct + ${formatNumber(judgedIncorrect)} judged incorrect + ${formatNumber(incomplete)} incomplete = ${formatNumber(accounted)}, while this benchmark records ${formatNumber(expected)} questions (${accountingDelta > 0 ? `${formatNumber(accountingDelta)} unaccounted` : `${formatNumber(Math.abs(accountingDelta))} extra`}).</span></aside>` : ""}`;
}

function renderEvaluationRecords(payload, append = false) {
  const records = payload.records || [];
  evaluationState.records = append ? evaluationState.records.concat(records) : records;
  evaluationState.page = payload.pagination.page;
  evaluationState.hasMore = payload.pagination.has_more;
  const counts = payload.counts || {};
  const judgedCorrect = counts.judged_correct ?? counts.correct ?? 0;
  const judgedIncorrect = counts.judged_incorrect ?? counts.incorrect ?? 0;
  elements["evaluation-result-count"].textContent = `${formatNumber(payload.pagination.total)} matching question${payload.pagination.total === 1 ? "" : "s"}`;
  elements["evaluation-result-caption"].textContent = `${formatNumber(judgedCorrect)} judged correct · ${formatNumber(judgedIncorrect)} judged incorrect · ${formatNumber(counts.incomplete)} incomplete`;
  elements["evaluation-records"].innerHTML = evaluationState.records.map((record) => `
    <button class="evaluation-record${record.query_id === evaluationState.selectedQueryId ? " selected" : ""}" data-evaluation-query="${escapeHtml(record.query_id)}">
      <span class="evaluation-record-top"><code>#${escapeHtml(record.query_id)}</code>${evaluationOutcomeBadge(record)}</span>
      <strong>${escapeHtml(record.question)}</strong>
      <span class="evaluation-record-response">${escapeHtml(record.response_preview || "No response")}</span>
      <span class="evaluation-record-meta">${formatNumber(record.tool_calls)} tool calls · ${formatNumber(record.step_count)} steps${record.completed ? "" : " · incomplete"}</span>
    </button>`).join("") || '<div class="evaluation-list-empty"><strong>No matching questions</strong><span>Try another outcome or search.</span></div>';
  elements["evaluation-load-more"].hidden = !evaluationState.hasMore;
  elements["evaluation-records"].querySelectorAll("[data-evaluation-query]").forEach((button) => {
    button.addEventListener("click", () => openEvaluationRecord(button.dataset.evaluationQuery));
  });
}

async function loadEvaluationList({ append = false } = {}) {
  if (append && evaluationState.loading) return null;
  evaluationState.loading = true;
  const requestVersion = ++evaluationState.listRequestVersion;
  const page = append ? evaluationState.page + 1 : 1;
  if (!append) elements["evaluation-records"].innerHTML = '<div class="catalog-loading"><i></i><span>Indexing judge results…</span></div>';
  try {
    const query = new URLSearchParams({
      outcome: evaluationState.outcome,
      search: evaluationState.search,
      sort: evaluationState.sort,
      page: String(page),
      page_size: "40",
    });
    const payload = await api(`/api/trainings/${encodeURIComponent(evaluationState.trainingId)}/evaluations/${encodeURIComponent(evaluationState.evaluationId)}?${query}`);
    if (requestVersion !== evaluationState.listRequestVersion) return null;
    renderEvaluationRecords(payload, append);
    return payload;
  } catch (error) {
    if (requestVersion !== evaluationState.listRequestVersion) return null;
    elements["evaluation-records"].innerHTML = `<div class="evaluation-list-empty"><strong>Evaluation could not be loaded</strong><span>${escapeHtml(error.message)}</span></div>`;
    showToast(error.message);
    return null;
  } finally {
    if (requestVersion === evaluationState.listRequestVersion) evaluationState.loading = false;
  }
}

function renderEvaluationDetail(payload) {
  const record = payload.record;
  const judge = payload.judge;
  const choices = payload.responses || [];
  const responseSelector = choices.length > 1 ? `<label class="evaluation-response-select"><span>Displayed rollout</span><select id="evaluation-response-select">${choices.map((choice) => `<option value="${choice.index}"${choice.index === payload.selected_response_index ? " selected" : ""}>${escapeHtml(choice.label)}${choice.judged ? " · judged" : ""}</option>`).join("")}</select></label>` : "";
  const counts = Object.entries(payload.kind_counts || {}).map(([kind, count]) => `<span class="turn-count" data-kind="${escapeHtml(kind)}"><strong>${formatNumber(count)}</strong>${escapeHtml(SEGMENT_LABELS[kind] || kind.replaceAll("_", " "))}</span>`).join("");
  const assistantTextNote = payload.kind_counts?.assistant_text
    ? '<span class="turns-summary-note">Assistant text means intermediate plain-text model messages; it is not the saved final response.</span>'
    : "";
  let judgeBody;
  if (!judge.applies_to_selected_response) {
    judgeBody = `<div class="evaluation-unjudged"><strong>This rollout was not passed to the evaluator.</strong><span>The correctness verdict belongs to rollout ${payload.judged_response_index + 1}; switch back to view its judge result.</span></div>`;
  } else if (record.outcome === "incomplete" || record.completed === false) {
    judgeBody = `<div class="evaluation-unjudged incomplete"><strong>The rollout was incomplete, so the LLM judge was not called.</strong><span>${escapeHtml(judge.error || "The stored terminal response did not satisfy the evaluator's completion rules.")}</span></div>`;
  } else if (record.outcome === "judged_correct" || record.outcome === "judged_incorrect" || typeof record.correct === "boolean") {
    judgeBody = `<div class="evaluation-judge-verdict">${evaluationOutcomeBadge(record)}<span>The verdict applies to this rollout.</span></div>
       <div class="evaluation-judge-grid"><section><h3>Judge explanation</h3><p>${escapeHtml(judge.reasoning || "No structured explanation was parsed.")}</p></section><section><h3>Extracted answer</h3><p>${escapeHtml(judge.extracted_final_answer || "Not extracted")}</p></section></div>
       <div class="evaluation-judge-raw-row"><details class="evaluation-raw-judge"><summary>Judge input</summary><pre>${escapeHtml(judge.prompt || "No judge prompt saved.")}</pre></details><details class="evaluation-raw-judge"><summary>Raw judge output</summary><pre>${escapeHtml(judge.output || "No raw judge output saved.")}</pre></details></div>`;
  } else {
    judgeBody = `<div class="evaluation-unjudged warning"><strong>This completed rollout has no usable judge verdict.</strong><span>${escapeHtml(judge.error || judge.parse_error || "Inspect the saved judge input and output for missing or unparseable results.")}</span></div>
      <div class="evaluation-judge-raw-row"><details class="evaluation-raw-judge"><summary>Judge input</summary><pre>${escapeHtml(judge.prompt || "No judge prompt saved.")}</pre></details><details class="evaluation-raw-judge"><summary>Raw judge output</summary><pre>${escapeHtml(judge.output || "No raw judge output saved.")}</pre></details></div>`;
  }
  elements["evaluation-detail"].innerHTML = `
    <header class="evaluation-detail-header"><div><p class="eyebrow">Question ${escapeHtml(record.query_id)}</p><h2>${escapeHtml(record.question)}</h2></div><div class="evaluation-detail-actions">${responseSelector}<button type="button" class="secondary-button rollout-locator-button" data-copy-evaluation-locator>Copy rollout locator</button></div></header>
    <div class="evaluation-answer-grid">
      <section><h3>Reference answer</h3><p>${escapeHtml(record.reference_answer || "Not recorded")}</p></section>
      <section><h3>Selected terminal response</h3><p>${escapeHtml(payload.response.terminal_text || "No terminal response")}</p></section>
      <section><h3>Rollout state</h3><p>${escapeHtml(payload.response.finish_reason || "unknown finish")} · ${escapeHtml(evaluationToolSummary(payload.response.tool_call_counts))}</p></section>
    </div>
    <section class="evaluation-judge-panel"><div class="section-title-row"><div><p class="eyebrow">Evaluator</p><h2>Judge result</h2></div></div>${judgeBody}</section>
    ${evaluationFinderPanel(payload, counts, assistantTextNote)}
    <section class="evaluation-transcript">${payload.segments.length ? payload.segments.map((segment, index, segments) => turnBlock(segment, precedingToolCall(segments, index), { allowFullDocument: true, responseIndex: payload.selected_response_index })).join("") : '<p class="muted">No structured trajectory messages were saved.</p>'}</section>`;
  wireTurnControls(elements["evaluation-detail"]);
  wireEvaluationFinder(elements["evaluation-detail"], payload);
  wireLocatorButton(
    elements["evaluation-detail"].querySelector("[data-copy-evaluation-locator]"),
    evaluationRolloutLocator(payload),
  );
  const selector = document.getElementById("evaluation-response-select");
  if (selector) selector.addEventListener("change", () => openEvaluationRecord(record.query_id, Number(selector.value)));
}

async function openEvaluationRecord(queryId, responseIndex = null) {
  closeFullDocument();
  evaluationState.selectedQueryId = queryId;
  elements["evaluation-records"].querySelectorAll("[data-evaluation-query]").forEach((button) => {
    button.classList.toggle("selected", button.dataset.evaluationQuery === queryId);
  });
  const requestVersion = ++evaluationState.detailRequestVersion;
  elements["evaluation-detail"]._evaluationFinderCleanup?.();
  elements["evaluation-detail"].innerHTML = '<div class="catalog-loading"><i></i><span>Loading raw inference trajectory…</span></div>';
  try {
    const suffix = responseIndex === null ? "" : `?response_index=${responseIndex}`;
    const payload = await api(`/api/trainings/${encodeURIComponent(evaluationState.trainingId)}/evaluations/${encodeURIComponent(evaluationState.evaluationId)}/records/${encodeURIComponent(queryId)}${suffix}`);
    if (requestVersion !== evaluationState.detailRequestVersion) return;
    history.replaceState(
      null,
      "",
      evaluationRolloutHash(
        evaluationState.trainingId,
        evaluationState.evaluationId,
        queryId,
        payload.selected_response_index,
      ),
    );
    renderEvaluationDetail(payload);
  } catch (error) {
    if (requestVersion !== evaluationState.detailRequestVersion) return;
    elements["evaluation-detail"].innerHTML = `<div class="evaluation-list-empty"><strong>Trajectory could not be loaded</strong><span>${escapeHtml(error.message)}</span></div>`;
  }
}

async function loadEvaluationExplorer(trainingId, evaluationId, focus = null) {
  const changed = evaluationState.trainingId !== trainingId || evaluationState.evaluationId !== evaluationId;
  evaluationState.trainingId = trainingId;
  evaluationState.evaluationId = evaluationId;
  if (changed) {
    evaluationState.listRequestVersion += 1;
    evaluationState.detailRequestVersion += 1;
    evaluationState.loading = false;
    evaluationState.outcome = "all";
    evaluationState.sort = "id";
    evaluationState.search = "";
    evaluationState.page = 0;
    evaluationState.records = [];
    evaluationState.selectedQueryId = null;
    elements["evaluation-outcome"].value = "all";
    elements["evaluation-sort"].value = "id";
    elements["evaluation-search"].value = "";
  }
  elements["evaluation-header"].innerHTML = '<div class="catalog-loading"><i></i><span>Loading evaluation provenance…</span></div>';
  const [trainingPayload, evaluationPayload] = await Promise.all([
    api(`/api/trainings/${encodeURIComponent(trainingId)}`),
    loadEvaluationList(),
  ]);
  if (!evaluationPayload) return;
  const training = trainingPayload.training || trainingPayload;
  renderEvaluationHeader(training, evaluationPayload);
  const first = evaluationState.records[0];
  if (focus?.queryId) await openEvaluationRecord(focus.queryId, focus.responseIndex);
  else if (first) await openEvaluationRecord(first.query_id);
  else elements["evaluation-detail"].innerHTML = '<div class="evaluation-list-empty"><strong>No trajectory selected</strong><span>Choose a result from the left.</span></div>';
}

function parseRoute() {
  const parts = (window.location.hash || "#/").replace(/^#\/?/, "").split("/").filter(Boolean);
  if (parts[0] !== "trainings" || !parts[1]) return { view: "overview", trainingId: null };
  let trainingId;
  try { trainingId = decodeURIComponent(parts[1]); } catch { trainingId = parts[1]; }
  if (parts[2] === "evaluations" && parts[3]) {
    let evaluationId;
    try { evaluationId = decodeURIComponent(parts[3]); } catch { evaluationId = parts[3]; }
    let evaluationFocus = null;
    if (parts[4] === "records" && parts[5]) {
      let queryId;
      try { queryId = decodeURIComponent(parts[5]); } catch { queryId = parts[5]; }
      const responseIndex = parts[6] === undefined ? null : Number(parts[6]);
      evaluationFocus = { queryId, responseIndex: Number.isInteger(responseIndex) ? responseIndex : null };
    }
    return { view: "evaluation", trainingId, evaluationId, evaluationFocus };
  }
  let groupFocus = null;
  if (parts[2] === "groups" && parts[3] && parts[4] && parts[5]) {
    let groupKey;
    try { groupKey = decodeURIComponent(parts[5]); } catch { groupKey = parts[5]; }
    groupFocus = { source: parts[3], step: Number(parts[4]), groupKey };
  }
  let rolloutFocus = null;
  if (parts[2] === "rollouts" && parts[3] && parts[4] && parts[5]) {
    let source;
    let recordId;
    try { source = decodeURIComponent(parts[3]); } catch { source = parts[3]; }
    try { recordId = decodeURIComponent(parts[5]); } catch { recordId = parts[5]; }
    rolloutFocus = { source, step: Number(parts[4]), recordId };
  }
  return {
    view: "training",
    trainingId,
    legacyRolloutRoute: parts[2] === "rollouts" && !rolloutFocus,
    groupFocus,
    rolloutFocus,
  };
}

function selectView(view) {
  if (view !== "evaluation") elements["evaluation-detail"]._evaluationFinderCleanup?.();
  elements["overview-panel"].hidden = view !== "overview";
  elements["training-panel"].hidden = view !== "training";
  elements["evaluation-panel"].hidden = view !== "evaluation";
  if (view === "overview") {
    elements["rollout-view"].hidden = true;
    elements["rollout-empty"].hidden = true;
  }
}

async function renderRoute() {
  closeDetail({ restoreRoute: false });
  closeGroupModal();
  clearTimeout(catalogState.refreshPoll);
  catalogState.refreshPoll = null;
  const route = parseRoute();
  selectView(route.view);
  window.scrollTo(0, 0);
  if (route.view === "overview") {
    catalogState.detailRequestVersion += 1;
    document.title = "Training Observatory";
    await loadCatalog();
    return;
  }
  if (route.view === "evaluation") {
    catalogState.detailRequestVersion += 1;
    document.title = "Evaluation · Training Observatory";
    await loadEvaluationExplorer(route.trainingId, route.evaluationId, route.evaluationFocus);
    return;
  }
  document.title = "Experiment · Training Observatory";
  catalogState.trainingId = route.trainingId;
  elements["inspector-back"].href = "#/";
  const training = await loadTrainingDetail(route.trainingId);
  if (!training) return;
  if (route.legacyRolloutRoute) {
    history.replaceState(null, "", `#/trainings/${encodeURIComponent(route.trainingId)}`);
  }
  const hasRollouts = rolloutAvailable(training);
  elements["rollout-empty"].hidden = hasRollouts;
  elements["rollout-view"].hidden = !hasRollouts;
  if (!hasRollouts) return;
  const targetRun = `training:${route.trainingId}`;
  try {
    await initialize({
      targetRun,
      preserve: state.inspectorInitialized && state.run === targetRun && !route.groupFocus && !route.rolloutFocus,
      targetSource: route.groupFocus?.source || route.rolloutFocus?.source,
      targetStep: route.groupFocus?.step ?? route.rolloutFocus?.step,
      targetGroupKey: route.groupFocus?.groupKey,
    });
    if (route.rolloutFocus?.recordId) await openDetail(route.rolloutFocus.recordId, { syncRoute: false });
  } catch (error) {
    showToast(error.message);
    elements["rollout-view"].hidden = true;
    elements["rollout-empty"].hidden = false;
    elements["rollout-empty"].querySelector("h2").textContent = "Rollouts could not be opened";
    elements["rollout-empty"].querySelector("p").textContent = error.message;
  }
}

async function initialize({
  preserve = false,
  targetRun = null,
  targetSource = null,
  targetStep = null,
  targetGroupKey = "",
} = {}) {
  const previousRun = state.run;
  const previousStep = state.step;
  state.meta = await api("/api/meta");
  state.categories = [
    { id: "all", label: "All trajectories", section: "Everything", description: "No review filter" },
    ...state.meta.categories,
  ];
  state.groupCategories = state.meta.group_categories || [];
  elements["root-path"].textContent = state.meta.root;
  elements["root-path"].title = state.meta.root;
  elements["run-select"].innerHTML = state.meta.runs
    .map((run) => {
      const suffix = run.attempts?.length > 1 ? ` · ${run.attempts.length} attempts` : "";
      return `<option value="${escapeHtml(run.name)}">${escapeHtml(run.label || run.name)}${escapeHtml(suffix)}</option>`;
    })
    .join("");
  const knownRuns = new Set(state.meta.runs.map((run) => run.name));
  state.run = targetRun && knownRuns.has(targetRun)
    ? targetRun
    : preserve && knownRuns.has(previousRun)
      ? previousRun
      : state.meta.default_run;
  if (!state.run) throw new Error("No rollout shards were discovered in this directory");
  if (targetRun && !knownRuns.has(targetRun)) {
    throw new Error("No indexed rollout artifacts are available for this training");
  }
  elements["run-select"].value = state.run;
  if (targetSource) state.source = targetSource;
  state.mode = "trajectories";
  state.category = targetGroupKey ? "all" : (preserve ? state.category : "review");
  state.groupKey = targetGroupKey;
  renderRunMeta();
  await loadSteps(targetStep ?? (preserve ? previousStep : null));
  renderViewMode();
  renderCategories();
  await loadPage(true);
  state.inspectorInitialized = true;
  elements["last-refresh"].textContent = `Refreshed ${new Date().toLocaleTimeString()}`;
}

function renderRunMeta() {
  const run = state.meta?.runs?.find((item) => item.name === state.run);
  if (!run) {
    elements["run-meta"].innerHTML = "";
    return;
  }
  const metadata = run.metadata || {};
  // Step ranges are resolved lazily, so prefer the loaded steps payload for this run.
  const current = state.stepsPayload?.run === state.run ? state.stepsPayload : run;
  const range = current.first_step === null || current.first_step === undefined
    ? "steps pending"
    : `steps ${current.first_step}–${current.last_step}`;
  const facts = [
    metadata.model_name ? `<span title="Model">${escapeHtml(metadata.model_name)}</span>` : "",
    metadata.git_commit ? `<span title="Git commit">@${escapeHtml(String(metadata.git_commit).slice(0, 8))}</span>` : "",
    `<span title="Artifact step range">${escapeHtml(range)}</span>`,
    `<span title="Saved rollout shards">${formatNumber(run.accepted_files)} learner-batch · ${formatNumber(run.filtered_files)} discarded-group shards</span>`,
    run.attempts?.length > 1 ? `<span title="Consolidated trainer starts">${formatNumber(run.attempts.length)} attempts</span>` : "",
  ].filter(Boolean);
  elements["run-meta"].innerHTML = facts.join("");
}

async function loadSteps(preferredStep = null) {
  const payload = await api(`/api/steps?run=${encodeURIComponent(state.run)}`);
  state.steps = payload.steps;
  state.stepsPayload = payload;
  state.sourceRanges = payload.source_ranges;
  const hasAccepted = Boolean(state.sourceRanges.accepted);
  const hasFiltered = Boolean(state.sourceRanges.filtered);
  if ((state.source === "accepted" && !hasAccepted)
      || (state.source === "filtered" && !hasFiltered)
      || (state.source === "both" && !(hasAccepted && hasFiltered))) {
    state.source = hasAccepted ? "accepted" : "filtered";
  }
  renderSourceControl();
  const sourceRange = state.source === "both"
    ? { first_step: payload.first_step, last_step: payload.last_step }
    : payload.source_ranges[state.source];
  const fallback = sourceRange?.last_step ?? payload.last_step;
  const hasPreferredStep = preferredStep !== null && preferredStep !== undefined && preferredStep !== "";
  const preferred = hasPreferredStep ? Number(preferredStep) : null;
  const preferredAvailable = hasPreferredStep && state.steps.includes(preferred)
    && (!sourceRange || (preferred >= sourceRange.first_step && preferred <= sourceRange.last_step));
  state.step = preferredAvailable ? preferred : fallback;
  renderStepControl();
  renderRunMeta();
}

function renderSourceControl() {
  const hasAccepted = Boolean(state.sourceRanges.accepted);
  const hasFiltered = Boolean(state.sourceRanges.filtered);
  elements["source-control"].querySelectorAll("button").forEach((button) => {
    const source = button.dataset.source;
    const available = source === "both" ? hasAccepted && hasFiltered : Boolean(state.sourceRanges[source]);
    button.disabled = !available;
    button.classList.toggle("active", source === state.source);
    button.setAttribute("aria-pressed", String(source === state.source));
    button.title = available
      ? SOURCE_DESCRIPTIONS[source]
      : source === "filtered"
        ? "No discarded-group rollout artifacts were saved for this run."
        : "This source is not available for this run.";
  });
  elements["source-help"].textContent = hasFiltered
    ? "Learner batch contains every trajectory from retained prompt groups, including reward 0. Discarded groups had zero reward variance and never reached the learner."
    : "Only learner-batch artifacts were saved for this run; active-sampling discards are unavailable.";
}

// Steps only exist for the source being viewed, so the scrubber spans that range.
function activeStepRange() {
  const payload = state.stepsPayload;
  if (!payload) return null;
  const whole = { first_step: payload.first_step, last_step: payload.last_step };
  if (state.source === "both") return whole;
  return payload.source_ranges?.[state.source] || whole;
}

// Steps carrying a validation score, when W&B could supply them.
function evaluatedStops() {
  const range = activeStepRange();
  const stops = state.stepsPayload?.evaluated_steps || [];
  if (!range || !stops.length) return null;
  const within = stops.filter((step) => step >= range.first_step && step <= range.last_step);
  return within.length ? within : null;
}

function validationAt(step) {
  return state.stepsPayload?.evaluations?.find((item) => item.artifact_step === Number(step)) || null;
}

function renderStepControl() {
  const range = activeStepRange();
  if (!range) return;
  const { first_step: first, last_step: last } = range;
  elements["step-input"].min = first;
  elements["step-input"].max = last;
  elements["step-input"].value = state.step;
  // This is an artifact-step scrubber, so its coordinate system must remain
  // the complete retained rollout range. Validation checkpoints annotate a
  // step but must not redefine the slider's maximum.
  const stops = evaluatedStops();
  const slider = elements["step-slider"];
  slider.min = first;
  slider.max = last;
  slider.value = state.step;
  slider.disabled = first === last;
  elements["step-range"].textContent = `${first}–${last}`;
  const scope = stops
    ? `${formatNumber(stops.length)} evaluated`
    : `${formatNumber(last - first + 1)} in ${sourceLabel(state.source)}`;
  const validation = validationAt(state.step);
  const score = validation ? ` · validation ${(validation.score * 100).toFixed(1)}%` : "";
  elements["step-caption"].textContent = `optimizer step ${state.step + 1} · ${scope}${score}`;
  elements["previous-step"].disabled = state.step <= first;
  elements["next-step"].disabled = state.step >= last;
  elements["step-latest"].disabled = state.step >= last;
}

function stepForSliderValue(value) {
  return Number(value);
}

function renderViewMode() {
  elements["view-mode-control"].querySelectorAll("button").forEach((button) => {
    const active = button.dataset.mode === state.mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  elements["view-mode-help"].textContent = state.mode === "groups"
    ? "Compare prompt difficulty and open every trajectory in a group together."
    : "Review individual sampled trajectories.";
  const options = state.mode === "groups"
    ? [
        ["reward", "Hardest pass rate"],
        ["suspicion", "Most flagged"],
        ["tokens", "Longest average"],
        ["calls", "Most tool calls"],
        ["sample", "Prompt order"],
      ]
    : [
        ["suspicion", "Most suspicious"],
        ["tokens", "Longest"],
        ["calls", "Most tool calls"],
        ["sample", "Prompt / sample"],
        ["reward", "Zero reward first"],
      ];
  elements["sort-select"].innerHTML = options
    .map(([value, label]) => `<option value="${value}">${label}</option>`)
    .join("");
  if (!options.some(([value]) => value === state.sort)) state.sort = options[0][0];
  elements["sort-select"].value = state.sort;
}

function renderCategories() {
  const categories = state.mode === "groups" ? state.groupCategories : state.categories;
  let section = null;
  elements["category-list"].innerHTML = categories
    .map((category) => {
      const count = state.categoryCounts[category.id];
      let heading = "";
      if (category.section && category.section !== section) {
        section = category.section;
        heading = `<p class="category-heading">${escapeHtml(section)}</p>`;
      }
      return `${heading}
        <button class="category-button ${category.id === state.category ? "active" : ""}"
          data-category="${escapeHtml(category.id)}" title="${escapeHtml(category.description)}">
          <span class="category-dot"></span>
          <span class="category-label">${escapeHtml(category.label)}</span>
          <span class="category-count">${count === undefined ? "" : formatNumber(count)}</span>
        </button>`;
    })
    .join("");
  elements["category-list"].querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.category = button.dataset.category;
      state.groupKey = "";
      renderCategories();
      loadPage(true);
    });
  });
}

function setStep(step) {
  const range = activeStepRange();
  const value = Number(step);
  if (!range || !Number.isFinite(value)) return;
  // Clamp rather than reject so dragging or typing past an end is not a dead end.
  state.step = Math.min(range.last_step, Math.max(range.first_step, Math.round(value)));
  state.groupKey = "";
  renderStepControl();
  loadPage(true);
}

function queryString(page) {
  const params = new URLSearchParams({
    run: state.run,
    step: state.step,
    source: state.source,
    category: state.category,
    sort: state.sort,
    search: state.search,
    group: state.groupKey,
    page,
    page_size: 24,
  });
  return params.toString();
}

async function loadPage(reset = false) {
  if (state.loading && !reset) return;
  const version = ++state.requestVersion;
  if (reset) {
    state.page = 0;
    state.hasMore = false;
    elements["record-grid"].innerHTML = skeletonCards(6);
    elements["empty-state"].hidden = true;
  }
  state.loading = true;
  elements["loading-label"].hidden = false;
  elements["load-more"].hidden = true;
  try {
    const nextPage = reset ? 1 : state.page + 1;
    const endpoint = state.mode === "groups" ? "/api/groups" : "/api/rollouts";
    const payload = await api(`${endpoint}?${queryString(nextPage)}`);
    if (version !== state.requestVersion) return;
    if (reset) elements["record-grid"].innerHTML = "";
    state.page = nextPage;
    state.hasMore = payload.has_more;
    state.categoryCounts = state.mode === "groups"
      ? { ...payload.category_counts, all_groups: payload.stats.groups }
      : { ...payload.category_counts, all: payload.stats.records };
    renderCategories();
    renderStats(payload);
    if (state.mode === "groups") {
      payload.groups.forEach((group) => elements["record-grid"].insertAdjacentHTML("beforeend", groupCard(group)));
      wireGroupCards();
    } else {
      payload.records.forEach((record) => elements["record-grid"].insertAdjacentHTML("beforeend", recordCard(record)));
      wireRecordCards();
    }
    elements["empty-state"].hidden = payload.total !== 0;
    elements["load-more"].hidden = !state.hasMore;
    const currentLabel = categoryLabels[state.category]
      || (state.mode === "groups" ? state.groupCategories : state.categories)
        .find((category) => category.id === state.category)?.label
      || state.category;
    elements["view-title"].textContent = currentLabel;
    elements["view-subtitle"].textContent =
      state.groupKey
        ? `All trajectories in group ${state.groupKey} · artifact step ${state.step}`
        : `Artifact step ${state.step} · optimizer step ${state.step + 1} · ${sourceLabel(state.source)}`;
  } catch (error) {
    if (version !== state.requestVersion) return;
    if (reset) elements["record-grid"].innerHTML = "";
    showToast(error.message);
  } finally {
    if (version === state.requestVersion) {
      state.loading = false;
      elements["loading-label"].hidden = true;
    }
  }
}

function renderStats(payload) {
  const groupMode = state.mode === "groups";
  elements["stat-records-label"].textContent = groupMode ? "Groups scanned" : "Rollouts scanned";
  elements["stat-reward-label"].textContent = groupMode ? "Mean group pass rate" : "Reward rate";
  elements["stat-records"].textContent = formatNumber(groupMode ? payload.stats.groups : payload.stats.records);
  const rewardRate = groupMode ? payload.stats.mean_group_pass_rate : payload.stats.reward_rate;
  elements["stat-reward"].textContent = rewardRate === null
    ? "—"
    : `${(rewardRate * 100).toFixed(1)}%`;
  elements["stat-tokens"].textContent = formatTokens(payload.stats.average_tokens);
  const validation = validationAt(payload.step);
  elements["stat-validation"].textContent = validation ? `${(validation.score * 100).toFixed(1)}%` : "—";
  elements["stat-validation"].title = validation
    ? `${validation.metric} · optimizer step ${validation.optimizer_step}`
    : "No validation was logged for this optimizer step";
  elements["stat-flagged"].textContent = formatNumber(payload.stats.suspicious);
  elements["result-count"].textContent = `${formatNumber(payload.total)} ${groupMode ? "groups" : "trajectories"}`;
}

function severity(record) {
  if (record.suspicion_score >= 9) return "high";
  if (record.suspicion_score >= 4) return "medium";
  return "low";
}

function groupNote(record, interactive = true) {
  if (!record.group_size) return "";
  const label = GROUP_DIFFICULTY_LABELS[record.group_difficulty]
    || GROUP_SHAPE_LABELS[record.group_shape]
    || "group";
  if (interactive && record.group_key) {
    return `<button type="button" class="group-inline-button" data-open-group data-group-source="${escapeHtml(record.source)}" data-group-step="${escapeHtml(record.step)}" data-group-key="${escapeHtml(record.group_key)}" title="Open ${escapeHtml(label)}">group <b>${formatNumber(record.group_correct)}/${formatNumber(record.group_size)}</b></button>`;
  }
  return `<span title="${escapeHtml(label)}">group <b>${formatNumber(record.group_correct)}/${formatNumber(record.group_size)}</b></span>`;
}

function recordCard(record, { groupAction = true } = {}) {
  const badges = record.categories
    .filter((category) => !HIDDEN_BADGES.has(category) && !category.startsWith("incomplete_"))
    .slice(0, 5)
    .map((category) => `<span class="badge ${escapeHtml(category)}">${escapeHtml(badgeLabel(category))}</span>`)
    .join("");
  const why = record.incomplete_reason
    ? `<span title="Why the verifier never scored it">${escapeHtml(record.incomplete_reason.replaceAll("_", " "))}</span>`
    : `<span><b>${formatNumber(record.successful_tool_calls)}</b>/${formatNumber(record.num_calls)} tools ok</span>`;
  const rewardValue = Number(record.reward || 0);
  const rewardTone = rewardValue > 0 ? "positive" : rewardValue < 0 ? "negative" : "zero";
  const verifierStatus = outcomeLabel(record.outcome);
  const verifierTitle = record.judged
    ? `This trajectory reached the verifier and was marked ${rewardValue > 0 ? "correct" : "incorrect"}.`
    : record.format_error_reason
      ? `Verifier skipped: ${record.format_error_reason.replaceAll("_", " ")}.`
      : record.incomplete_reason
        ? `Verifier skipped: ${record.incomplete_reason.replaceAll("_", " ")}.`
        : "This trajectory did not reach the verifier.";
  const cardLabel = `Prompt ${record.prompt_idx}, sample ${record.sample_idx}, ${sourceLabel(record.source)}, reward ${formatNumber(rewardValue, 3)}, ${verifierStatus}`;
  return `
    <article class="record-card" data-id="${escapeHtml(record.id)}" data-severity="${severity(record)}" role="button" aria-label="${escapeHtml(cardLabel)}" tabindex="0">
      <div class="card-top">
        <span class="card-index">P${record.prompt_idx} · S${record.sample_idx} · ${escapeHtml(sourceLabel(record.source))}</span>
        <span class="card-status">
          <span class="reward-pill ${rewardTone}">Reward ${formatNumber(rewardValue, 3)}</span>
          <span class="outcome-pill ${escapeHtml(record.outcome || "")}" title="${escapeHtml(verifierTitle)}">${escapeHtml(verifierStatus)}</span>
        </span>
      </div>
      <h3>${escapeHtml(record.ground_truth || "Unknown reference")}</h3>
      <p class="preview">${escapeHtml(record.terminal_preview || "No terminal response captured.")}</p>
      <div class="metric-row">
        <span><b>${formatTokens(record.token_count)}</b> tokens</span>
        ${why}
        ${groupNote(record, groupAction)}
        <span><b>${formatNumber(record.answer_declaration_count)}</b> answers</span>
      </div>
      <div class="badge-row">${badges}</div>
    </article>`;
}

function groupCard(group) {
  const rate = Number(group.pass_rate || 0);
  const difficulty = GROUP_DIFFICULTY_LABELS[group.difficulty] || group.difficulty;
  return `
    <article class="record-card group-card" data-group-key="${escapeHtml(group.group_key)}" data-group-source="${escapeHtml(group.source)}" data-group-step="${escapeHtml(group.step)}" role="button" tabindex="0" aria-label="Open prompt group with ${formatNumber(group.correct)} of ${formatNumber(group.size)} passing trajectories">
      <div class="card-top">
        <span class="card-index">${escapeHtml(sourceLabel(group.source))} · ${escapeHtml(group.prompt_id || `prompt ${group.prompt_idx}`)}</span>
        <span class="badge ${escapeHtml(group.difficulty)}" title="${escapeHtml(difficulty)}">${escapeHtml(group.difficulty.replaceAll("_", " "))}</span>
      </div>
      <h3>${escapeHtml(group.ground_truth || "Unknown reference")}</h3>
      <div class="group-pass-row"><strong>${(rate * 100).toFixed(1)}%</strong><span>${formatNumber(group.correct)}/${formatNumber(group.size)} passed</span></div>
      <div class="group-pass-track" aria-label="${(rate * 100).toFixed(1)} percent pass rate"><i style="width:${Math.max(0, Math.min(100, rate * 100))}%"></i></div>
      <p class="preview">${escapeHtml(group.terminal_preview || "No terminal response preview captured.")}</p>
      <div class="metric-row">
        <span><b>${formatTokens(group.average_tokens)}</b> avg tokens</span>
        <span><b>${formatNumber(group.average_calls, 1)}</b> avg calls</span>
        <span><b>${formatNumber(group.token_capped)}</b> capped</span>
        <span><b>${formatNumber(group.format_errors)}</b> format errors</span>
        <span><b>${formatNumber(group.incomplete)}</b> incomplete</span>
      </div>
    </article>`;
}

function skeletonCards(count) {
  return Array.from({ length: count }, (_, index) => `
    <article class="record-card skeleton" style="opacity:${0.76 - index * 0.06}">
      <div class="card-index">Indexing one requested step…</div>
      <h3>Reading record summaries</h3>
      <p class="preview">Large token and logprob arrays remain on disk.</p>
    </article>`).join("");
}

function wireRecordCards(root = elements["record-grid"]) {
  root.querySelectorAll(".record-card[data-id]").forEach((card) => {
    card.onclick = () => openDetail(card.dataset.id);
    card.onkeydown = (event) => {
      if (event.target !== card) return;
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openDetail(card.dataset.id);
      }
    };
  });
  root.querySelectorAll("[data-open-group]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      openGroup(button.dataset.groupSource, Number(button.dataset.groupStep), button.dataset.groupKey);
    });
  });
}

function wireGroupCards() {
  elements["record-grid"].querySelectorAll(".group-card[data-group-key]").forEach((card) => {
    const open = () => openGroup(card.dataset.groupSource, Number(card.dataset.groupStep), card.dataset.groupKey);
    card.addEventListener("click", open);
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open();
      }
    });
  });
}

async function openGroup(source, step, groupKey) {
  const identity = `${source}:${step}:${groupKey}`;
  const version = ++state.groupModalRequestVersion;
  state.groupModalKey = identity;
  elements["group-modal-backdrop"].hidden = false;
  elements["group-modal"].classList.add("open");
  elements["group-modal"].setAttribute("aria-hidden", "false");
  elements["group-modal-kicker"].textContent = `Artifact ${step} · optimizer step ${step + 1} · ${sourceLabel(source)}`;
  elements["group-modal-title"].textContent = "Loading prompt group…";
  elements["group-modal-summary"].innerHTML = "";
  elements["group-modal-body"].innerHTML = '<div class="detail-loading">Loading every trajectory in this group…</div>';
  elements["group-modal-close"].focus();
  const params = new URLSearchParams({
    run: state.run,
    step,
    source,
    category: "all",
    sort: "sample",
    search: "",
    group: groupKey,
    page: 1,
    page_size: 256,
  });
  try {
    const payload = await api(`/api/rollouts?${params}`);
    if (version !== state.groupModalRequestVersion || state.groupModalKey !== identity) return;
    renderGroupModal(payload);
  } catch (error) {
    if (version !== state.groupModalRequestVersion || state.groupModalKey !== identity) return;
    elements["group-modal-title"].textContent = "Group could not be loaded";
    elements["group-modal-body"].innerHTML = `<div class="empty-state"><h3>Could not load prompt group</h3><p>${escapeHtml(error.message)}</p></div>`;
  }
}

function renderGroupModal(payload) {
  const records = payload.records || [];
  if (!records.length) {
    elements["group-modal-title"].textContent = "Empty prompt group";
    elements["group-modal-body"].innerHTML = '<div class="empty-state"><h3>No trajectories found</h3><p>This group is no longer present in the selected rollout artifact.</p></div>';
    return;
  }
  const first = records[0];
  const size = Number(first.group_size || records.length);
  const correct = Number(first.group_correct ?? records.filter((record) => Number(record.reward || 0) > 0).length);
  const passRate = size ? correct / size : 0;
  const difficulty = GROUP_DIFFICULTY_LABELS[first.group_difficulty]
    || GROUP_SHAPE_LABELS[first.group_shape]
    || "Prompt group";
  const averageTokens = records.reduce((total, record) => total + Number(record.token_count || 0), 0) / records.length;
  const averageCalls = records.reduce((total, record) => total + Number(record.num_calls || 0), 0) / records.length;
  const capped = records.filter((record) => record.categories.includes("token_capped")).length;
  const formatErrors = records.filter((record) => record.categories.includes("format_error")).length;
  const incomplete = records.filter((record) => record.outcome === "incomplete").length;
  elements["group-modal-title"].textContent = first.ground_truth || "Unknown reference";
  elements["group-modal-summary"].innerHTML = `
    <span class="badge ${escapeHtml(first.group_difficulty)}">${escapeHtml(difficulty)}</span>
    <span><b>${formatNumber(correct)}/${formatNumber(size)}</b> passed · <b>${(passRate * 100).toFixed(1)}%</b></span>
    <span><b>${formatTokens(averageTokens)}</b> average length</span>
    <span><b>${formatNumber(averageCalls, 1)}</b> average tool calls</span>
    <span><b>${formatNumber(capped)}</b> capped</span>
    <span><b>${formatNumber(formatErrors)}</b> format errors</span>
    <span><b>${formatNumber(incomplete)}</b> incomplete</span>`;
  elements["group-modal-body"].innerHTML = `<div class="group-modal-grid">${records.map((record) => recordCard(record, { groupAction: false })).join("")}</div>`;
  wireRecordCards(elements["group-modal-body"]);
}

function closeGroupModal() {
  state.groupModalKey = null;
  state.groupModalRequestVersion += 1;
  elements["group-modal"].classList.remove("open");
  elements["group-modal"].setAttribute("aria-hidden", "true");
  setTimeout(() => {
    if (!state.groupModalKey) elements["group-modal-backdrop"].hidden = true;
  }, 220);
}

async function openDetail(recordId, { syncRoute = true } = {}) {
  document.getElementById("turns-body")?._evaluationFinderCleanup?.();
  state.detailId = recordId;
  state.traceOffset = 0;
  elements["drawer-backdrop"].hidden = false;
  elements["detail-drawer"].classList.add("open");
  elements["detail-drawer"].setAttribute("aria-hidden", "false");
  elements["detail-title"].textContent = "Loading trajectory…";
  elements["detail-copy-locator"].hidden = true;
  elements["detail-copy-locator"].disabled = true;
  elements["detail-body"].innerHTML = '<div class="detail-loading">Reading one record from its byte offset…</div>';
  try {
    const detail = await api(`/api/rollouts/${encodeURIComponent(recordId)}`);
    if (state.detailId !== recordId) return;
    if (syncRoute && catalogState.trainingId) {
      const hash = trainingRolloutHash(catalogState.trainingId, detail);
      if (window.location.hash !== hash) history.pushState(null, "", hash);
    }
    renderDetail(detail);
  } catch (error) {
    elements["detail-body"].innerHTML = `<div class="empty-state"><h3>Could not load trajectory</h3><p>${escapeHtml(error.message)}</p></div>`;
  }
}

function renderDetail(detail) {
  elements["detail-kicker"].textContent = `Artifact ${detail.step} · prompt ${detail.prompt_idx} · sample ${detail.sample_idx}`;
  elements["detail-title"].textContent = detail.ground_truth || "Unknown reference";
  wireLocatorButton(
    elements["detail-copy-locator"],
    trainingRolloutLocator(catalogState.trainingId || "unknown-training", detail),
  );
  const reasons = detail.reasons.length
    ? `<ul class="reason-list">${detail.reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}</ul>`
    : '<p class="muted">No structural warning detected.</p>';
  const answers = detail.answer_declarations.length
    ? `<div class="answer-list">${detail.answer_declarations.map((answer) => `<div class="answer-item">${escapeHtml(answer)}</div>`).join("")}</div>`
    : '<p class="muted">No Answer:/Final Answer: or XML answer declaration detected.</p>';
  const references = detail.ground_truths?.length ? detail.ground_truths : [detail.ground_truth].filter(Boolean);
  const reference = references.length
    ? `<div class="answer-list">${references.map((value) => `<div class="answer-item reference-item">${escapeHtml(value)}</div>`).join("")}</div>
       <p class="muted">${detail.ground_truth_mentioned
         ? "An exact-normalized reference answer appears in the verifier-visible response."
         : "No exact-normalized reference answer appears in the verifier-visible response."}</p>`
    : '<p class="muted">This rollout stored no reference answer.</p>';
  const prompt = detail.raw_prompt
    ? `<section class="detail-section"><h3>Question / prompt</h3><pre class="code-panel">${escapeHtml(detail.raw_prompt)}</pre></section>`
    : "";
  const decoded = detail.decoded_response_preview
    ? `<section class="detail-section"><h3>Saved decoded response${detail.decoded_response_truncated ? " · preview" : ""}</h3><pre class="code-panel">${escapeHtml(detail.decoded_response_preview)}</pre></section>`
    : "";
  const toolErrors = detail.tool_errors
    ? `<section class="detail-section"><h3>Tool errors${detail.tool_errors_truncated ? " · preview" : ""}</h3><pre class="code-panel">${escapeHtml(detail.tool_errors)}</pre></section>`
    : "";
  const toolOutputs = detail.tool_outputs
    ? `<section class="detail-section"><h3>Tool outputs${detail.tool_outputs_truncated ? " · preview" : ""}</h3><pre class="code-panel">${escapeHtml(detail.tool_outputs)}</pre></section>`
    : "";
  const rawTerminal = `<section class="detail-section"><h3>Raw terminal model turn</h3><pre class="code-panel prose">${escapeHtml(detail.terminal_response || "No terminal model turn was captured.")}</pre></section>`;
  const reconstructedVerifierInput = detail.verifier_input_source === "reconstructed";
  const reconstructedVerifierSkip = detail.verifier_skipped_reason_source === "reconstructed";
  const verifierInput = detail.verifier_input
    ? `<section class="detail-section"><h3>Verifier input · ${reconstructedVerifierInput ? "reconstructed at display time" : "exact saved prompt"}</h3><pre class="code-panel prose">${escapeHtml(detail.verifier_input)}</pre>${reconstructedVerifierInput ? '<p class="muted">Derived in memory with the current verifier logic from this rollout’s stored question, reference answer, terminal turn, and registered format-gate setting. The artifact was not changed.</p>' : ""}</section>`
    : detail.verifier_skipped_reason
      ? `<section class="detail-section"><h3>Verifier skipped${reconstructedVerifierSkip ? " · reconstructed" : ""}</h3><p class="preview">No judge request was made: <code>${escapeHtml(detail.verifier_skipped_reason)}</code></p>${reconstructedVerifierSkip ? '<p class="muted">Derived at display time; the rollout artifact was not changed.</p>' : ""}</section>`
      : `<section class="detail-section"><h3>Verifier input</h3><p class="muted">Not recorded for this historical rollout.</p></section>`;
  const judgeOutput = detail.judge_output
    ? `<section class="detail-section"><h3>Judge output · saved attempts</h3><pre class="code-panel prose">${escapeHtml(JSON.stringify(detail.judge_output, null, 2))}</pre></section>`
    : `<section class="detail-section"><h3>Judge output</h3><p class="muted">No detailed judge output was recorded.</p></section>`;
  elements["detail-body"].innerHTML = `
    <div class="detail-layout">
      <aside class="detail-rail">
        <div class="detail-meta">
          <span>Outcome <b>${escapeHtml(outcomeLabel(detail.outcome))}</b></span>
          <span>Reward <b>${formatNumber(detail.reward, 3)}</b></span>
          <span>Advantage <b>${formatNumber(detail.advantage, 3)}</b></span>
          ${detail.group_size ? `<span title="${escapeHtml(GROUP_SHAPE_LABELS[detail.group_shape] || "")}">Group <b>${formatNumber(detail.group_correct)}/${formatNumber(detail.group_size)} correct</b></span>` : ""}
          <span>Length <b>${formatNumber(detail.token_count)}</b></span>
          <span>Tools <b>${formatNumber(detail.successful_tool_calls)}/${formatNumber(detail.num_calls)} ok</b></span>
          <span>Finish <b>${escapeHtml(detail.finish_reason)}</b></span>
          ${detail.termination_reason ? `<span>Termination <b>${escapeHtml(detail.termination_reason)}</b></span>` : ""}
          <span>Source <b>${escapeHtml(sourceLabel(detail.source))}</b></span>
        </div>
        ${detail.group_key ? `<button type="button" class="group-inline-button drawer-group-button" data-open-detail-group data-group-source="${escapeHtml(detail.source)}" data-group-step="${escapeHtml(detail.step)}" data-group-key="${escapeHtml(detail.group_key)}">View all ${formatNumber(detail.group_size)} trajectories in this group</button>` : ""}
        <div class="badge-row">${detail.categories.map((category) => `<span class="badge ${escapeHtml(category)}">${escapeHtml(badgeLabel(category))}</span>`).join("")}</div>
        <section class="detail-section"><h3>Reference answer · ground truth</h3>${reference}</section>
        <section class="detail-section"><h3>What the model answered</h3>${answers}</section>
        <section class="detail-section"><h3>Why it was surfaced</h3>${reasons}</section>
        ${prompt}
        ${toolErrors}
        ${toolOutputs}
        ${rawTerminal}
        ${verifierInput}
        ${judgeOutput}
        ${decoded}
        ${detail.trace_available ? `
          <section class="detail-section" id="trace-section">
            <h3>Full token trace · decoded on demand</h3>
            <div class="trace-actions"><button id="trace-load" class="secondary-button">Decode first 50k characters</button></div>
            <p class="trace-note">The only action that materializes response token IDs, paged in 50k-character chunks.</p>
            <pre id="trace-content" class="code-panel" hidden></pre>
          </section>` : ""}
      </aside>
      <div class="detail-main">
        <div id="turns-body" class="turns-body"><p class="muted">Building turns…</p></div>
      </div>
    </div>
  `;
  const groupButton = elements["detail-body"].querySelector("[data-open-detail-group]");
  if (groupButton) {
    groupButton.addEventListener("click", () => {
      const source = groupButton.dataset.groupSource;
      const step = Number(groupButton.dataset.groupStep);
      const groupKey = groupButton.dataset.groupKey;
      closeDetail();
      openGroup(source, step, groupKey);
    });
  }
  const traceButton = document.getElementById("trace-load");
  if (traceButton) traceButton.addEventListener("click", loadTrace);
  loadTurns(detail.id);
}

function safeHttpUrl(value) {
  try {
    const url = new URL(String(value || "").trim());
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : "";
  } catch {
    return "";
  }
}

function visitUrlFromToolCall(segment) {
  if (!segment || segment.kind !== "tool_call") return "";
  const toolName = String(segment.tool_name || "").toLowerCase();
  const content = String(segment.content || "");
  try {
    const argumentsObject = JSON.parse(content);
    const parsedUrl = safeHttpUrl(argumentsObject?.url);
    if (parsedUrl && (!toolName || toolName === "visit")) return parsedUrl;
  } catch {
    // Some saved tool calls use Qwen XML or a Bash command instead of JSON.
  }
  if (toolName === "visit" || !toolName) {
    const directUrl = content.match(/https?:\/\/[^\s"'<>}]+/i)?.[0];
    if (directUrl) return safeHttpUrl(directUrl);
  }
  if (toolName === "bash") {
    const visitCommand = content.match(/\bvisit\s+(?:"([^"]+)"|'([^']+)'|(https?:\/\/[^\s;&|]+))/i);
    return safeHttpUrl(visitCommand?.[1] || visitCommand?.[2] || visitCommand?.[3]);
  }
  return "";
}

function precedingToolCall(segments, index) {
  for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
    if (segments[cursor].kind === "tool_call") return segments[cursor];
    if (segments[cursor].kind === "tool_result") return null;
  }
  return null;
}

function turnBlock(segment, previousSegment = null, options = {}) {
  const label = SEGMENT_LABELS[segment.kind] || segment.kind.replaceAll("_", " ");
  const title = segment.tool_name ? `${label}: ${segment.tool_name}` : label;
  const note = segment.truncated
    ? `first ${formatNumber(segment.content.length)} of ${formatNumber(segment.char_len)} chars`
    : `${formatNumber(segment.char_len)} chars`;
  const documentCounts = segment.document_match_counts || {};
  const evidenceMarkers = Array.from({ length: Number(documentCounts.evidence || 0) }, () => '<span class="document-marker evidence" title="Evidence document" aria-label="Evidence document">●</span>').join("");
  const goldMarkers = Array.from({ length: Number(documentCounts.positive || 0) }, () => '<span class="document-marker gold" title="Gold document" aria-label="Gold document">★</span>').join("");
  const documentMarkers = evidenceMarkers || goldMarkers ? `<span class="document-markers" aria-label="BrowseComp-Plus relevant documents">${evidenceMarkers}${goldMarkers}</span>` : "";
  const documentKinds = `${Number(documentCounts.evidence || 0) > 0 ? " has-evidence-document" : ""}${Number(documentCounts.positive || 0) > 0 ? " has-gold-document" : ""}`;
  const isNotFound = segment.kind === "tool_result" && /\b404\s+not\s+found\b/i.test(String(segment.content || ""));
  const visitUrl = isNotFound ? visitUrlFromToolCall(previousSegment) : "";
  const notFoundBadge = isNotFound ? `<span class="turn-error-actions"><span class="turn-error-badge">(404 not found)</span>${visitUrl ? `<a class="turn-visit-link" data-visit-site href="${escapeHtml(visitUrl)}" target="_blank" rel="noopener noreferrer">Visit site ↗</a>` : ""}</span>` : "";
  const sourceUrl = visitUrl || visitUrlFromToolCall(previousSegment);
  const isVisitResult = segment.kind === "tool_result" && (
    String(segment.tool_name || previousSegment?.tool_name || "").toLowerCase() === "visit"
    || (String(previousSegment?.tool_name || "").toLowerCase() === "bash" && Boolean(sourceUrl))
  );
  const fullDocumentButton = options.allowFullDocument && isVisitResult && segment.truncated
    ? `<button type="button" class="turn-full-document" data-full-document="${segment.index}" data-response-index="${Number(options.responseIndex)}" data-visit-url="${escapeHtml(sourceUrl)}">Open full document</button>`
    : "";
  return `
    <details class="turn-block${documentKinds}" data-kind="${escapeHtml(segment.kind)}" data-segment-index="${segment.index}"${COLLAPSED_KINDS.has(segment.kind) ? "" : " open"}>
      <summary>
        <span class="turn-index">${segment.index + 1}</span>
        <span class="turn-label">${escapeHtml(title)}</span>
        ${fullDocumentButton}
        ${notFoundBadge}
        ${documentMarkers}
        <span class="turn-note">${escapeHtml(note)}</span>
      </summary>
      <pre class="turn-content ${PROSE_KINDS.has(segment.kind) ? "prose" : ""}">${highlightedSegmentText(segment)}</pre>
    </details>`;
}

function normalizeFullDocumentResult(result) {
  const matches = (result?.matches || []).map((match, index) => ({
    ...match,
    in_preview: true,
    match_index: index,
  }));
  return {
    ...(result || {}),
    total: Number(result?.total ?? matches.length),
    returned: matches.length,
    truncated: Boolean(result?.truncated),
    matches,
  };
}

function literalFullDocumentResult(content, term, segmentIndex) {
  const needle = String(term || "").trim();
  if (!needle) return normalizeFullDocumentResult({ term: "", total: 0, matches: [] });
  const foldedContent = content.toLocaleLowerCase();
  const foldedNeedle = needle.toLocaleLowerCase();
  const matches = [];
  let total = 0;
  let cursor = 0;
  while (cursor <= foldedContent.length - foldedNeedle.length) {
    const start = foldedContent.indexOf(foldedNeedle, cursor);
    if (start < 0) break;
    total += 1;
    if (matches.length < 1_000) {
      matches.push({
        segment_index: segmentIndex,
        segment_kind: "tool_result",
        category: "tool_result",
        start,
        end: start + needle.length,
        in_preview: true,
      });
    }
    cursor = start + Math.max(foldedNeedle.length, 1);
  }
  return normalizeFullDocumentResult({
    term: needle,
    total,
    matches,
    truncated: total > matches.length,
  });
}

function fullDocumentCountLabel(result, index, emptyLabel) {
  const total = Number(result?.total || 0);
  const available = result?.matches?.length || 0;
  if (!total) return emptyLabel;
  if (index >= 0 && result.truncated) return `${formatNumber(index + 1)} / ${formatNumber(available)} · ${formatNumber(total)} total`;
  if (index >= 0) return `${formatNumber(index + 1)} / ${formatNumber(total)}`;
  if (result.truncated) return `${formatNumber(total)} matches · first ${formatNumber(available)} navigable`;
  return `${formatNumber(total)} match${total === 1 ? "" : "es"}`;
}

function paintFullDocument(result, source, currentIndex = -1) {
  const payload = fullDocumentState.payload;
  if (!payload) return;
  const segment = payload.segment;
  elements["full-document-body"].innerHTML = `<pre class="full-document-content">${highlightedSegmentText(segment, result?.matches || [], currentIndex, source)}</pre>`;
}

function updateFullDocumentNavigation() {
  const source = fullDocumentState.activeSource;
  const result = source === "search" ? fullDocumentState.searchResult : fullDocumentState.referenceResult;
  const index = fullDocumentState.currentIndex;
  const referenceOutput = elements["full-document-toolbar"].querySelector('[data-full-document-count="reference"]');
  const searchOutput = elements["full-document-toolbar"].querySelector('[data-full-document-count="search"]');
  if (referenceOutput) {
    referenceOutput.textContent = fullDocumentCountLabel(
      fullDocumentState.referenceResult,
      source === "reference" ? index : -1,
      "0 matches",
    );
  }
  const searchInput = elements["full-document-toolbar"].querySelector("[data-full-document-search]");
  if (searchOutput) {
    searchOutput.textContent = fullDocumentCountLabel(
      fullDocumentState.searchResult,
      source === "search" ? index : -1,
      searchInput?.value.trim() ? "No matches" : "Enter text",
    );
  }
  ["reference", "search"].forEach((kind) => {
    const candidate = kind === "reference" ? fullDocumentState.referenceResult : fullDocumentState.searchResult;
    elements["full-document-toolbar"].querySelectorAll(`[data-full-document-nav^="${kind}-"]`).forEach((button) => {
      button.disabled = !(candidate?.matches?.length);
    });
  });
  paintFullDocument(result, source, index);
}

function activateFullDocumentMatches(source, result, index = -1) {
  fullDocumentState.activeSource = source;
  if (source === "search") fullDocumentState.searchResult = result;
  else fullDocumentState.referenceResult = result;
  fullDocumentState.currentIndex = index;
  updateFullDocumentNavigation();
}

function revealFullDocumentMatch(delta) {
  const result = fullDocumentState.activeSource === "search"
    ? fullDocumentState.searchResult
    : fullDocumentState.referenceResult;
  const available = result?.matches?.length || 0;
  if (!available) return;
  const previous = fullDocumentState.currentIndex;
  fullDocumentState.currentIndex = previous < 0
    ? (delta > 0 ? 0 : available - 1)
    : (previous + delta + available) % available;
  updateFullDocumentNavigation();
  elements["full-document-body"].querySelector(`.trajectory-match[data-match-index="${fullDocumentState.currentIndex}"]`)?.scrollIntoView({
    behavior: "smooth",
    block: "center",
  });
}

function wireFullDocumentToolbar() {
  const toolbar = elements["full-document-toolbar"];
  toolbar.querySelector("[data-full-document-reference]")?.addEventListener("click", () => {
    activateFullDocumentMatches("reference", fullDocumentState.referenceResult);
  });
  toolbar.querySelectorAll("[data-full-document-nav]").forEach((button) => {
    button.addEventListener("click", () => {
      const [source, direction] = button.dataset.fullDocumentNav.split("-");
      fullDocumentState.activeSource = source;
      revealFullDocumentMatch(direction === "next" ? 1 : -1);
    });
  });
  const input = toolbar.querySelector("[data-full-document-search]");
  input?.addEventListener("input", () => {
    clearTimeout(fullDocumentState.searchDebounce);
    fullDocumentState.searchDebounce = setTimeout(() => {
      const result = literalFullDocumentResult(
        fullDocumentState.payload?.segment?.content || "",
        input.value,
        fullDocumentState.payload?.segment?.index ?? 0,
      );
      if (input.value.trim()) activateFullDocumentMatches("search", result);
      else activateFullDocumentMatches("reference", fullDocumentState.referenceResult);
    }, 120);
  });
}

function renderFullDocument(payload, sourceUrl) {
  fullDocumentState.payload = payload;
  fullDocumentState.referenceResult = normalizeFullDocumentResult(payload.reference_matches);
  fullDocumentState.searchResult = null;
  fullDocumentState.currentIndex = -1;
  const segment = payload.segment;
  const regions = segment.document_regions || [];
  const relevance = regions.some((region) => region.kind === "gold")
    ? { kind: "gold", symbol: "★", label: "Gold source" }
    : regions.some((region) => region.kind === "evidence")
      ? { kind: "evidence", symbol: "●", label: "Evidence source" }
      : { kind: "", symbol: "", label: "Visit response" };
  const referenceTerm = payload.reference_matches?.term || "";
  elements["full-document-title"].textContent = `Visit document · ${formatNumber(segment.char_len)} characters`;
  if (sourceUrl) {
    elements["full-document-source"].href = sourceUrl;
    elements["full-document-source"].textContent = sourceUrl;
    elements["full-document-source"].hidden = false;
  } else {
    elements["full-document-source"].hidden = true;
  }
  elements["full-document-toolbar"].innerHTML = `
    <div class="full-document-reference">
      <button type="button" class="term" data-full-document-reference${referenceTerm ? "" : " disabled"} title="Highlight reference answer">Reference: ${escapeHtml(referenceTerm || "Not recorded")}</button>
      <div class="evaluation-find-nav">
        <button type="button" class="icon-button" data-full-document-nav="reference-prev" aria-label="Previous reference match">↑</button>
        <button type="button" class="icon-button" data-full-document-nav="reference-next" aria-label="Next reference match">↓</button>
        <output data-full-document-count="reference"></output>
      </div>
    </div>
    <div class="full-document-search">
      <input type="search" data-full-document-search maxlength="256" placeholder="Search complete document…" autocomplete="off" spellcheck="false" />
      <div class="evaluation-find-nav">
        <button type="button" class="icon-button" data-full-document-nav="search-prev" aria-label="Previous document search match" disabled>↑</button>
        <button type="button" class="icon-button" data-full-document-nav="search-next" aria-label="Next document search match" disabled>↓</button>
        <output data-full-document-count="search">Enter text</output>
      </div>
    </div>
    <span class="full-document-relevance ${relevance.kind}">${relevance.symbol ? `<span aria-hidden="true">${relevance.symbol}</span>` : ""}${escapeHtml(relevance.label)}</span>`;
  wireFullDocumentToolbar();
  const existingSearch = elements["evaluation-detail"].querySelector("[data-trajectory-search]")?.value.trim() || "";
  if (existingSearch) {
    const input = elements["full-document-toolbar"].querySelector("[data-full-document-search]");
    input.value = existingSearch;
    activateFullDocumentMatches(
      "search",
      literalFullDocumentResult(segment.content, existingSearch, segment.index),
    );
  } else {
    activateFullDocumentMatches("reference", fullDocumentState.referenceResult);
  }
}

async function openFullDocument(segmentIndex, responseIndex, sourceUrl, trigger) {
  const requestVersion = ++fullDocumentState.requestVersion;
  fullDocumentState.returnFocus = trigger;
  elements["full-document-backdrop"].hidden = false;
  elements["full-document-pane"].classList.add("open");
  elements["full-document-pane"].setAttribute("aria-hidden", "false");
  elements["full-document-title"].textContent = "Loading complete visit response…";
  elements["full-document-source"].hidden = true;
  elements["full-document-toolbar"].innerHTML = "";
  elements["full-document-body"].innerHTML = '<div class="detail-loading">Loading complete document…</div>';
  elements["full-document-close"].focus();
  try {
    const params = new URLSearchParams({ response_index: String(responseIndex) });
    const payload = await api(`/api/trainings/${encodeURIComponent(evaluationState.trainingId)}/evaluations/${encodeURIComponent(evaluationState.evaluationId)}/records/${encodeURIComponent(evaluationState.selectedQueryId)}/segments/${segmentIndex}?${params}`);
    if (requestVersion !== fullDocumentState.requestVersion) return;
    renderFullDocument(payload, sourceUrl);
  } catch (error) {
    if (requestVersion !== fullDocumentState.requestVersion) return;
    elements["full-document-title"].textContent = "Document could not be loaded";
    elements["full-document-body"].innerHTML = `<div class="empty-state"><h3>Could not load complete visit response</h3><p>${escapeHtml(error.message)}</p></div>`;
  }
}

function closeFullDocument() {
  if (!elements["full-document-pane"].classList.contains("open")) return;
  fullDocumentState.requestVersion += 1;
  clearTimeout(fullDocumentState.searchDebounce);
  elements["full-document-pane"].classList.remove("open");
  elements["full-document-pane"].setAttribute("aria-hidden", "true");
  setTimeout(() => { elements["full-document-backdrop"].hidden = true; }, 200);
  fullDocumentState.returnFocus?.focus();
  fullDocumentState.returnFocus = null;
  fullDocumentState.payload = null;
}

function wireTurnControls(host) {
  host.querySelectorAll("[data-full-document]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      openFullDocument(
        Number(button.dataset.fullDocument),
        Number(button.dataset.responseIndex),
        button.dataset.visitUrl,
        button,
      );
    });
  });
  host.querySelectorAll("[data-visit-site]").forEach((link) => {
    link.addEventListener("click", (event) => event.stopPropagation());
  });
  host.querySelectorAll("[data-turns-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const blocks = host.querySelectorAll(".turn-block");
      if (button.dataset.turnsAction === "expand") blocks.forEach((block) => { block.open = true; });
      if (button.dataset.turnsAction === "collapse") blocks.forEach((block) => { block.open = false; });
      if (button.dataset.turnsAction === "tools") {
        const hidden = host.classList.toggle("hide-tools");
        button.textContent = hidden ? "Show tool results" : "Hide tool results";
        button.classList.toggle("active", hidden);
      }
    });
  });
}

async function loadTurns(recordId) {
  const host = document.getElementById("turns-body");
  if (!host) return;
  try {
    const payload = await api(`/api/rollouts/${encodeURIComponent(recordId)}/turns`);
    if (state.detailId !== recordId || !document.getElementById("turns-body")) return;
    if (!payload.segments.length) {
      host.innerHTML = '<p class="muted">No turns could be parsed from this trajectory.</p>';
      return;
    }
    const counts = Object.entries(payload.kind_counts || {})
      .map(([kind, count]) => `<span class="turn-count" data-kind="${escapeHtml(kind)}"><strong>${formatNumber(count)}</strong>${escapeHtml(SEGMENT_LABELS[kind] || kind.replaceAll("_", " "))}</span>`)
      .join("");
    const assistantTextNote = payload.kind_counts?.assistant_text
      ? '<span class="turns-summary-note">Assistant text means intermediate plain-text model messages; the final one is shown separately as Final response.</span>'
      : "";
    host.innerHTML = `
      ${evaluationFinderPanel(payload, counts, assistantTextNote)}
      <section class="evaluation-transcript">${payload.segments.map((segment, index, segments) => turnBlock(segment, precedingToolCall(segments, index))).join("")}</section>`;
    wireTurnControls(host);
    wireEvaluationFinder(host, payload, {
      searchMatches: (query, signal) => api(
        `/api/rollouts/${encodeURIComponent(recordId)}/matches?${new URLSearchParams({ query })}`,
        { signal },
      ),
    });
  } catch (error) {
    if (state.detailId !== recordId) return;
    host.innerHTML = `<p class="muted">Could not build turns: ${escapeHtml(error.message)}</p>`;
  }
}

async function loadTrace() {
  const button = document.getElementById("trace-load");
  const panel = document.getElementById("trace-content");
  if (!button || !panel || !state.detailId) return;
  button.disabled = true;
  button.textContent = state.traceOffset ? "Loading next chunk…" : "Decoding trace…";
  try {
    const payload = await api(`/api/rollouts/${encodeURIComponent(state.detailId)}/trace?offset=${state.traceOffset}&limit=50000`);
    panel.hidden = false;
    panel.textContent += payload.content;
    state.traceOffset += payload.content.length;
    button.hidden = !payload.has_more;
    button.textContent = `Load next 50k · ${formatNumber(state.traceOffset)} / ${formatNumber(payload.total_chars)}`;
  } catch (error) {
    showToast(error.message);
    button.textContent = "Retry trace decoding";
  } finally {
    button.disabled = false;
  }
}

function closeDetail({ restoreRoute = true } = {}) {
  document.getElementById("turns-body")?._evaluationFinderCleanup?.();
  state.detailId = null;
  elements["detail-copy-locator"].hidden = true;
  elements["detail-copy-locator"].onclick = null;
  elements["detail-drawer"].classList.remove("open");
  elements["detail-drawer"].setAttribute("aria-hidden", "true");
  setTimeout(() => { elements["drawer-backdrop"].hidden = true; }, 220);
  if (restoreRoute) {
    const route = parseRoute();
    if (route.view === "training" && route.rolloutFocus) {
      history.replaceState(null, "", `#/trainings/${encodeURIComponent(route.trainingId)}`);
    }
  }
}

elements["run-select"].addEventListener("change", async () => {
  state.run = elements["run-select"].value;
  state.groupKey = "";
  const selected = state.meta?.runs?.find((item) => item.name === state.run);
  if (selected?.registry_id) {
    window.location.hash = `#/trainings/${encodeURIComponent(selected.registry_id)}`;
    return;
  }
  renderRunMeta();
  await loadSteps();
  loadPage(true);
});
elements["step-input"].addEventListener("change", () => setStep(elements["step-input"].value));
elements["previous-step"].addEventListener("click", () => setStep(state.step - 1));
elements["next-step"].addEventListener("click", () => setStep(state.step + 1));
elements["step-latest"].addEventListener("click", () => setStep(activeStepRange()?.last_step ?? state.step));
elements["view-mode-control"].querySelectorAll("button").forEach((button) => {
  button.addEventListener("click", () => {
    const mode = button.dataset.mode;
    if (mode === state.mode) return;
    state.mode = mode;
    state.groupKey = "";
    state.category = mode === "groups" ? "all_groups" : "review";
    state.sort = mode === "groups" ? "reward" : "suspicion";
    state.categoryCounts = {};
    renderViewMode();
    renderCategories();
    loadPage(true);
  });
});
// Dragging previews the target step; the fetch waits for release so scrubbing
// across a long run does not queue a request per pixel.
elements["step-slider"].addEventListener("input", () => {
  const step = stepForSliderValue(elements["step-slider"].value);
  elements["step-input"].value = step;
  const validation = validationAt(step);
  const score = validation ? ` · validation ${(validation.score * 100).toFixed(1)}%` : "";
  elements["step-caption"].textContent = `optimizer step ${step + 1}${score} · release to load`;
});
elements["step-slider"].addEventListener("change", () => setStep(stepForSliderValue(elements["step-slider"].value)));
elements["source-control"].querySelectorAll("button").forEach((button) => {
  button.addEventListener("click", () => {
    if (button.disabled) return;
    state.source = button.dataset.source;
    state.groupKey = "";
    renderSourceControl();
    // Re-clamps into the new source's range, then reloads.
    setStep(state.step);
  });
});
elements["sort-select"].addEventListener("change", () => {
  state.sort = elements["sort-select"].value;
  loadPage(true);
});
elements["search-input"].addEventListener("input", () => {
  clearTimeout(elements["search-input"].debounce);
  elements["search-input"].debounce = setTimeout(() => {
    state.search = elements["search-input"].value.trim();
    loadPage(true);
  }, 280);
});
elements["refresh-button"].addEventListener("click", async () => {
  elements["refresh-button"].disabled = true;
  elements["refresh-button"].textContent = "Refreshing…";
  try {
    await api("/api/refresh", { method: "POST" });
    await initialize({ preserve: true });
  } catch (error) {
    showToast(error.message);
  } finally {
    elements["refresh-button"].disabled = false;
    elements["refresh-button"].textContent = "↻ Refresh files";
  }
});
elements["load-more"].addEventListener("click", () => loadPage(false));
elements["group-modal-close"].addEventListener("click", closeGroupModal);
elements["group-modal-backdrop"].addEventListener("click", closeGroupModal);
elements["drawer-close"].addEventListener("click", closeDetail);
elements["drawer-backdrop"].addEventListener("click", closeDetail);
elements["full-document-close"].addEventListener("click", closeFullDocument);
elements["full-document-backdrop"].addEventListener("click", closeFullDocument);
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (elements["full-document-pane"].classList.contains("open")) closeFullDocument();
  else if (state.detailId) closeDetail();
  else if (state.groupModalKey) closeGroupModal();
});

elements["catalog-search"].addEventListener("input", () => {
  clearTimeout(elements["catalog-search"].debounce);
  elements["catalog-search"].debounce = setTimeout(() => {
    catalogState.search = elements["catalog-search"].value.trim();
    renderTrainingTable();
  }, 160);
});
elements["catalog-classification"].addEventListener("change", () => {
  catalogState.classification = elements["catalog-classification"].value;
  renderTrainingTable();
});
elements["catalog-sort"].addEventListener("change", () => {
  catalogState.sort = elements["catalog-sort"].value;
  renderTrainingTable();
});
elements["catalog-archived"].addEventListener("change", () => {
  catalogState.includeArchived = elements["catalog-archived"].checked;
  renderTrainingTable();
});
elements["catalog-smoke"].addEventListener("change", () => {
  catalogState.includeSmoke = elements["catalog-smoke"].checked;
  renderTrainingTable();
});
elements["catalog-refresh"].addEventListener("click", async () => {
  elements["catalog-refresh"].disabled = true;
  elements["catalog-refresh"].textContent = "Refreshing…";
  try {
    await api("/api/refresh", { method: "POST" });
    catalogState.metricsByTraining = {};
    await loadCatalog({ force: true });
  } catch (error) {
    showToast(error.message);
  } finally {
    elements["catalog-refresh"].disabled = false;
    elements["catalog-refresh"].textContent = "Refresh live data";
  }
});

elements["evaluation-outcome"].addEventListener("change", async () => {
  evaluationState.outcome = elements["evaluation-outcome"].value;
  evaluationState.selectedQueryId = null;
  const payload = await loadEvaluationList();
  const first = payload?.records?.[0];
  if (first) openEvaluationRecord(first.query_id);
  else elements["evaluation-detail"].innerHTML = '<div class="evaluation-list-empty"><strong>No matching trajectories</strong><span>Try another outcome or search.</span></div>';
});
elements["evaluation-sort"].addEventListener("change", () => {
  evaluationState.sort = elements["evaluation-sort"].value;
  loadEvaluationList();
});
elements["evaluation-search"].addEventListener("input", () => {
  clearTimeout(elements["evaluation-search"].debounce);
  elements["evaluation-search"].debounce = setTimeout(async () => {
    evaluationState.search = elements["evaluation-search"].value.trim();
    evaluationState.selectedQueryId = null;
    const payload = await loadEvaluationList();
    const first = payload?.records?.[0];
    if (first) openEvaluationRecord(first.query_id);
    else elements["evaluation-detail"].innerHTML = '<div class="evaluation-list-empty"><strong>No matching trajectories</strong><span>Try another outcome or search.</span></div>';
  }, 240);
});
elements["evaluation-load-more"].addEventListener("click", () => loadEvaluationList({ append: true }));

const observer = new IntersectionObserver((entries) => {
  if (entries[0]?.isIntersecting && state.hasMore && !state.loading) loadPage(false);
}, { rootMargin: "500px" });
observer.observe(elements["load-sentinel"]);

window.addEventListener("hashchange", () => {
  renderRoute().catch((error) => showToast(error.message));
});

renderRoute().catch((error) => {
  showToast(error.message);
  elements["training-table"].innerHTML = `<div class="empty-state"><h3>Viewer could not start</h3><p>${escapeHtml(error.message)}</p></div>`;
});
