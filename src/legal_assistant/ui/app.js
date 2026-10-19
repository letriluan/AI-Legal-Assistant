"use strict";
const $ = id => document.getElementById(id);
let busy = false, answerId = 0, appMode = "preview";
const emptyTemplate = $("empty").cloneNode(true);

function element(tag, text, cls) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

function controls(value) {
  busy = value;
  $("question").disabled = value;
  $("clear").disabled = value;
  document.querySelectorAll(".sample").forEach(button => button.disabled = value);
  $("send").disabled = value || !$("question").value.trim();
  $("send").textContent = value ? "Synthesizing Brief…" : "Analyze & Cite →";
}

function message(role, text) {
  $("empty")?.remove();
  // In a Legal Dossier workbench, each new inquiry replaces the active brief cleanly
  if (role === "user") {
    $("chat").replaceChildren();
  }
  const article = element("article", "", "message " + role);
  article.append(
    element("strong", role === "user" ? "ACTIVE STATUTORY INQUIRY" : "LEGAL SYNTHESIS & STATUTORY CITATIONS"),
    element("p", text)
  );
  $("chat").append(article);
  return article;
}

function renderResponse(article, data) {
  article.replaceChildren();

  // Create 2-column Legal Dossier layout: Left = Legal Opinion Brief, Right = Statutory Evidence Cards
  const grid = element("div", "", "dossier-grid");
  const opinionCol = element("div", "", "opinion-col");
  const evidenceCol = element("aside", "", "evidence-col");

  // Header inside Opinion column
  const opinionHeader = element("div", "", "opinion-header");
  opinionHeader.append(
    element("strong", "SYNTHESIZED LEGAL OPINION"),
    element("span", data.mode === "preview" ? "ARCHIVED SUBMISSION OUTPUT" : "LIVE RAG · " + Math.round(data.latency_ms) + " MS", "tag")
  );
  opinionCol.append(opinionHeader);

  let shownText = data.answer;
  const bilingual = data.mode === "preview" && Boolean(data.answer_en);

  if (bilingual) {
    const prefix = "answer-" + (++answerId);
    const tabs = element("div", "", "answer-tabs");
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", "Answer language");

    const notice = element("p", data.translation_notice, "translation-note");
    const panel = element("div", "", "answer-panel");
    panel.id = prefix + "-panel";
    panel.setAttribute("role", "tabpanel");
    panel.tabIndex = 0;

    const content = element("p", "", "opinion-body");
    panel.append(content);

    const choices = [
      { id: "en", label: "English Legal Brief", text: data.answer_en },
      { id: "vi", label: "Original Vietnamese (Gốc)", text: data.answer }
    ];

    const buttons = choices.map(choice => {
      const button = element("button", choice.label);
      button.type = "button";
      button.id = prefix + "-" + choice.id;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-controls", panel.id);
      tabs.append(button);
      return button;
    });

    function select(index, focus = false) {
      choices.forEach((choice, i) => {
        buttons[i].setAttribute("aria-selected", String(index === i));
        buttons[i].tabIndex = index === i ? 0 : -1;
      });
      panel.setAttribute("aria-labelledby", buttons[index].id);
      panel.lang = choices[index].id;
      content.textContent = choices[index].text;
      shownText = choices[index].text;
      notice.textContent = index === 0
        ? data.translation_notice
        : "Original Vietnamese output retained from the submission; not newly generated.";
      if (focus) buttons[index].focus();
    }

    buttons.forEach((button, index) => {
      button.addEventListener("click", () => select(index));
      button.addEventListener("keydown", event => {
        if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
          event.preventDefault();
          select(event.key === "Home" ? 0 : event.key === "End" ? 1 : 1 - index, true);
        }
      });
    });

    select(0);
    opinionCol.append(tabs, notice, panel);
  } else {
    const answer = element("p", data.answer, "opinion-body");
    if (data.mode === "rag") answer.lang = "vi";
    opinionCol.append(answer);
    if (data.mode === "rag") {
      opinionCol.append(
        element("p", "Live answers are generated in Vietnamese. Prepared English translations are available only for saved preview examples.", "translation-note")
      );
    }
  }

  if (data.review_note) {
    opinionCol.append(element("p", data.review_note, "review-note"));
  }

  const footerBar = element("div", "", "opinion-footer");
  const copy = element("button", "Copy Legal Opinion", "copy-answer");
  copy.type = "button";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(shownText);
      copy.textContent = "Copied to Clipboard ✓";
    } catch {
      copy.textContent = "Copy unavailable";
    }
    setTimeout(() => copy.textContent = "Copy Legal Opinion", 1800);
  });
  footerBar.append(copy);
  opinionCol.append(footerBar);

  // Populate Right Column: Statutory Citations & Evidence
  const sources = data.sources || [];
  const citeHeader = element("div", "", "evidence-header");
  citeHeader.append(
    element("h3", "CITED STATUTORY AUTHORITIES (" + sources.length + ")", "source-heading"),
    element("span", "Selected via RRF + Cross-Encoder + Veto Rules", "evidence-subtitle")
  );
  evidenceCol.append(citeHeader);

  if (!sources.length) {
    evidenceCol.append(element("p", "No statutory citations attached to this response.", "no-sources"));
  }

  sources.forEach((source, idx) => {
    const details = element("details", "", "source");
    details.open = true; // Open by default in the Dossier view so citations stand out immediately!
    const summaryTitle = [
      source.article_en || source.article,
      source.document_en || source.document,
      source.code
    ].filter(Boolean).join(" · ");

    const sumEl = element("summary", "");
    sumEl.append(
      element("span", "EXHIBIT 0" + (idx + 1), "exhibit-badge"),
      element("span", summaryTitle, "exhibit-title")
    );
    details.append(sumEl);

    const bodyWrap = element("div", "", "source-body");
    const original = element("p", [source.article, source.document, source.code].filter(Boolean).join(" · "), "original-source");
    original.lang = "vi";
    bodyWrap.append(
      element("span", "OFFICIAL VIETNAMESE STATUTE REFERENCE", "source-kicker"),
      original,
      element("p", source.excerpt, "source-excerpt")
    );
    if (source.document_en) {
      bodyWrap.append(element("p", "English document titles are descriptive translations, not official legal translations.", "source-translation-note"));
    }
    details.append(bodyWrap);
    evidenceCol.append(details);
  });

  grid.append(opinionCol, evidenceCol);
  article.append(grid);
}

$("question").addEventListener("input", () => controls(busy));

$("clear").addEventListener("click", () => {
  $("chat").replaceChildren(emptyTemplate.cloneNode(true));
  $("question").value = "";
  document.querySelectorAll(".sample").forEach(sample => sample.setAttribute("aria-pressed", "false"));
  controls(false);
});

$("form").addEventListener("submit", async event => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question || busy) return;
  message("user", question);
  controls(true);
  const article = message("assistant", "Retrieving statutory articles & synthesizing legal brief…");
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: AbortSignal.timeout(300000)
    });
    if (!response.ok) {
      throw new Error(
        response.status === 429
          ? "Another request is running. Please retry shortly."
          : "Service unavailable. Check the preview database or local RAG setup."
      );
    }
    renderResponse(article, await response.json());
  } catch (error) {
    article.replaceChildren(element("p", error.message, "error"));
  } finally {
    controls(false);
  }
});

async function initialize() {
  try {
    const response = await fetch("/examples");
    if (!response.ok) throw new Error();
    const data = await response.json();
    appMode = data.mode;
    $("mode").textContent = appMode === "preview" ? "Bilingual Archive Mode (2,000 cases)" : "Live Hybrid RAG Mode";
    $("banner").textContent = appMode === "preview"
      ? "2,000 archived submission answers · 6 bilingual briefs with English translations · Exact match replay"
      : "Live RAG active · English sample cards map to Vietnamese statutory queries";

    if (appMode === "preview") {
      const library = element("details", "", "archive-library");
      library.append(element("summary", "🔍 Search Full 2,000-Question Vietnamese Statutory Archive (Click to expand)"));
      const input = element("input");
      input.type = "search";
      input.placeholder = "Type a Vietnamese legal keyword, e.g. doanh nghiep, bao hiem, hop dong, thue…";
      input.setAttribute("aria-label", "Search archived questions");
      const results = element("div", "", "archive-results");
      library.append(input, results);
      $("examples").before(library);

      let timer, revision = 0;
      async function search() {
        const version = ++revision;
        try {
          const response = await fetch("/archive/questions?q=" + encodeURIComponent(input.value));
          if (!response.ok) throw new Error();
          const payload = await response.json();
          if (version !== revision) return;
          results.replaceChildren();
          for (const row of payload.questions) {
            const button = element("button", row.question, "archive-question");
            button.type = "button";
            button.lang = "vi";
            button.addEventListener("click", () => {
              if (!busy) {
                $("question").value = row.question;
                controls(false);
                $("form").requestSubmit();
              }
            });
            results.append(button);
          }
          if (!payload.questions.length) results.append(element("p", "No archived questions found."));
        } catch {
          results.textContent = "Question library unavailable.";
        }
      }
      input.addEventListener("input", () => {
        clearTimeout(timer);
        revision++;
        timer = setTimeout(search, 250);
      });
      library.addEventListener("toggle", () => {
        if (library.open) search();
      });
    }

    data.examples.forEach((example, index) => {
      const button = element("button", "", "sample");
      button.type = "button";
      button.setAttribute("aria-pressed", "false");

      const topRow = element("div", "", "sample-top");
      topRow.append(
        element("span", "CASE " + String(index + 1).padStart(2, "0"), "sample-number"),
        element("span", example.topic_en, "sample-topic"),
        element("span", "Inspect Brief →", "sample-arrow")
      );

      const label = element("div", "", "sample-label");
      label.append(element("span", example.question_en, "sample-question-en"));
      const vietnamese = element("span", example.question, "sample-vi");
      vietnamese.lang = "vi";
      label.append(vietnamese);

      button.append(topRow, label);
      button.addEventListener("click", () => {
        if (!busy) {
          document.querySelectorAll(".sample").forEach(sample => sample.setAttribute("aria-pressed", String(sample === button)));
          $("question").value = example.question_en;
          controls(false);
          $("form").requestSubmit();
        }
      });
      $("examples").append(button);
    });
  } catch {
    $("mode").textContent = "Backend unavailable";
  }
}

initialize();
