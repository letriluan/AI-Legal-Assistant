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
  $("send").textContent = value ? "Searching…" : "Search Law";
}

function message(role, text) {
  $("empty")?.remove();
  if (role === "user") {
    $("chat").replaceChildren();
  }
  const article = element("article", "", "message " + role);
  article.append(
    element("strong", role === "user" ? "Question" : "Answer"),
    element("p", text)
  );
  $("chat").append(article);
  return article;
}

function renderResponse(article, data) {
  article.replaceChildren();

  const header = element("div", "", "answer-header");
  header.append(
    element("strong", "Answer"),
    element("span", data.mode === "preview" ? "Archived response" : "Live RAG · " + Math.round(data.latency_ms) + " ms", "tag")
  );
  article.append(header);

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

    const content = element("p");
    panel.append(content);

    const choices = [
      { id: "en", label: "English translation", text: data.answer_en },
      { id: "vi", label: "Original Vietnamese", text: data.answer }
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
        : "Original Vietnamese output retained from the submission.";
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
    article.append(tabs, notice, panel);
  } else {
    const answer = element("p", data.answer);
    if (data.mode === "rag") answer.lang = "vi";
    article.append(answer);
    if (data.mode === "rag") {
      article.append(
        element("p", "Live answers are generated in Vietnamese. Prepared English translations are available only for saved preview examples.", "translation-note")
      );
    }
  }

  const copy = element("button", "Copy answer", "copy-answer");
  copy.type = "button";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(shownText);
      copy.textContent = "Copied ✓";
    } catch {
      copy.textContent = "Copy unavailable";
    }
    setTimeout(() => copy.textContent = "Copy answer", 1800);
  });
  article.append(copy);

  if (data.review_note) {
    article.append(element("p", data.review_note, "review-note"));
  }

  const sources = data.sources || [];
  if (sources.length) {
    article.append(element("h3", "Cited Legal Articles (" + sources.length + ")", "source-heading"));
  }

  for (const source of sources) {
    const details = element("details", "", "source");
    details.open = true;
    details.append(
      element("summary", [source.article_en || source.article, source.document_en || source.document, source.code].filter(Boolean).join(" · "))
    );
    const original = element("p", [source.article, source.document, source.code].filter(Boolean).join(" · "), "original-source");
    original.lang = "vi";
    details.append(
      element("span", "Original Vietnamese reference:", "source-kicker"),
      original,
      element("p", source.excerpt)
    );
    article.append(details);
  }
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
  const article = message("assistant", "Loading…");
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
    $("mode").textContent = appMode === "preview" ? "Preview Mode (2,000 saved answers)" : "Full RAG Mode";
    $("banner").textContent = appMode === "preview"
      ? "Preview mode: 2,000 archived answers in SQLite (6 include English translations). No live LLM cost."
      : "Full RAG mode: English sample questions map to their original Vietnamese queries.";

    if (appMode === "preview") {
      const library = element("details", "", "archive-library");
      library.append(element("summary", "Browse all 2,000 archived Vietnamese questions"));
      const input = element("input");
      input.type = "search";
      input.placeholder = "Filter by keyword (e.g. bao hiem, hop dong, thue)…";
      input.setAttribute("aria-label", "Search archived questions");
      const results = element("div");
      library.append(input, results);
      $("examples").after(library);

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
      const label = element("span", "", "sample-label");
      label.append(
        element("span", (index + 1) + ". " + example.topic_en, "sample-topic"),
        element("span", example.question_en, "sample-question-en")
      );
      const vietnamese = element("span", example.question, "sample-vi");
      vietnamese.lang = "vi";
      label.append(vietnamese);
      button.append(label);
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
