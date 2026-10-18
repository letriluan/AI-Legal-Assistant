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
 busy = value; $("question").disabled = value; $("clear").disabled = value;
 document.querySelectorAll(".sample").forEach(button => button.disabled = value);
 $("send").disabled = value || !$("question").value.trim();
 $("send").textContent = value ? "Working…" : "Ask question ↗";
}
function message(role, text) {
 $("empty")?.remove(); const article = element("article", "", "message " + role);
 article.append(element("strong", role === "user" ? "Your question" : "Research response"), element("p",text));
 $("chat").append(article); return article;
}
function renderResponse(article, data) {
 article.replaceChildren(element("strong","Research response"));
 let shownText = data.answer;
 const bilingual = data.mode === "preview" && Boolean(data.answer_en);
 if (bilingual) {
  const prefix = "answer-" + (++answerId);
  const tabs = element("div", "", "answer-tabs"); tabs.setAttribute("role","tablist"); tabs.setAttribute("aria-label","Answer language");
  const notice = element("p", data.translation_notice, "translation-note");
  const panel = element("div", "", "answer-panel"); panel.id = prefix + "-panel"; panel.setAttribute("role","tabpanel"); panel.tabIndex = 0;
  const content = element("p"); panel.append(content);
  const choices = [{id:"en",label:"English translation",text:data.answer_en}, {id:"vi",label:"Original Vietnamese",text:data.answer}];
  const buttons = choices.map(choice => {
   const button = element("button",choice.label); button.type="button"; button.id=prefix+"-"+choice.id;
   button.setAttribute("role","tab"); button.setAttribute("aria-controls",panel.id); tabs.append(button); return button;
  });
  function select(index, focus=false) {
   choices.forEach((choice,i) => { buttons[i].setAttribute("aria-selected",String(index===i)); buttons[i].tabIndex=index===i?0:-1; });
   panel.setAttribute("aria-labelledby",buttons[index].id); panel.lang=choices[index].id; content.textContent=choices[index].text; shownText=choices[index].text;
   notice.textContent=index===0?data.translation_notice:"Original Vietnamese output retained from the submission; not newly generated.";
   if (focus) buttons[index].focus();
  }
  buttons.forEach((button,index) => {
   button.addEventListener("click",()=>select(index));
   button.addEventListener("keydown",event=>{ if (["ArrowLeft","ArrowRight","Home","End"].includes(event.key)) { event.preventDefault(); select(event.key==="Home"?0:event.key==="End"?1:1-index,true); } });
  });
  select(0); article.append(tabs,notice,panel);
 } else {
  const answer = element("p",data.answer); if(data.mode==="rag") answer.lang="vi"; article.append(answer);
  if(data.mode==="rag") article.append(element("p","Live answers are generated in Vietnamese. Prepared English translations are available only for saved preview examples.","translation-note"));
 }
 const copy=element("button","Copy displayed answer","copy-answer"); copy.type="button";
 copy.addEventListener("click",async()=>{try{await navigator.clipboard.writeText(shownText);copy.textContent="Copied";}catch{copy.textContent="Copy unavailable";} setTimeout(()=>copy.textContent="Copy displayed answer",1800);});
 article.append(copy,element("div",data.mode==="preview"?"Archived response · not newly generated":"Live RAG response · "+Math.round(data.latency_ms)+" ms","tag"));
 if(data.review_note) article.append(element("p",data.review_note,"review-note"));
 const sources=data.sources||[];
 if(sources.length) article.append(element("h3","DOCUMENT REFERENCES","source-heading"));
 for(const source of sources) {
  const details=element("details","","source");
  details.append(element("summary",[source.article_en||source.article,source.document_en||source.document,source.code].filter(Boolean).join(" · ")));
  const original=element("p",[source.article,source.document,source.code].filter(Boolean).join(" · "),"original-source"); original.lang="vi";
  details.append(element("span","ORIGINAL REFERENCE","source-kicker"),original,element("p",source.excerpt));
  if(source.document_en) details.append(element("p","English document titles are descriptive translations, not official legal translations.","source-translation-note"));
  article.append(details);
 }
}
$("question").addEventListener("input",()=>controls(busy));
$("clear").addEventListener("click",()=>{
 $("chat").replaceChildren(emptyTemplate.cloneNode(true)); $("question").value="";
 document.querySelectorAll(".sample").forEach(sample=>sample.setAttribute("aria-pressed","false")); controls(false);
});
$("form").addEventListener("submit",async event=>{
 event.preventDefault(); const question=$("question").value.trim(); if(!question||busy)return;
 message("user",question); controls(true); const article=message("assistant","Loading…");
 try {
  const response=await fetch("/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question}),signal:AbortSignal.timeout(300000)});
  if(!response.ok)throw new Error(response.status===429?"Another request is running. Please retry shortly.":"Full RAG is unavailable. Check local artifacts, models and Ollama.");
  renderResponse(article,await response.json());
 } catch(error) {article.replaceChildren(element("p",error.message,"error"));}
 finally {controls(false);$("chat").scrollTop=$("chat").scrollHeight;}
});
async function initialize(){
 try {
  const response=await fetch("/examples"); if(!response.ok)throw new Error(); const data=await response.json(); appMode=data.mode;
  $("mode").textContent=appMode==="preview"?"Bilingual preview — no model calls":"Full RAG — Vietnamese generation";
  $("banner").textContent=appMode==="preview"?"Saved Vietnamese responses with prepared English translations. No new answers are generated.":"English example questions map to Vietnamese originals. Live answers are not automatically translated.";
  data.examples.forEach((example,index)=>{
   const button=element("button","","sample");button.type="button";button.setAttribute("aria-pressed","false");
   const label=element("span","","sample-label");
   label.append(element("span",example.topic_en,"sample-topic"),element("span",example.question_en,"sample-question-en"));
   const vietnamese=element("span",example.question,"sample-vi");vietnamese.lang="vi";label.append(vietnamese);
   button.append(element("span",String(index+1).padStart(2,"0"),"sample-number"),label,element("span","↗","sample-arrow"));
   button.addEventListener("click",()=>{if(!busy){document.querySelectorAll(".sample").forEach(sample=>sample.setAttribute("aria-pressed",String(sample===button)));$("question").value=example.question_en;controls(false);$("question").focus();}});
   $("examples").append(button);
  });
 }catch{$("mode").textContent="Backend unavailable";}
}
initialize();
