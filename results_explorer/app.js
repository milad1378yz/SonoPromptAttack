const state={examples:[],visible:[],selected:null,timer:null};
const $=id=>document.getElementById(id);
const unique=(items,key)=>[...new Set(items.map(item=>item[key]))].sort();
const fill=(select,values,current)=>{select.innerHTML=values.map(v=>`<option${v===current?" selected":""}>${v}</option>`).join("")};
const formatTask=value=>value==="DD"?"Disease diagnosis":value==="LL"?"Lesion localization":value;

function updateFilters(origin){
  let rows=state.examples;
  if(origin!=="proposer"&&$("proposer").value) rows=rows.filter(x=>x.proposer_model===$("proposer").value);
  fill($("proposer"),unique(state.examples,"proposer_model"),$("proposer").value);
  rows=state.examples.filter(x=>x.proposer_model===$("proposer").value);
  fill($("target"),unique(rows,"target_vlm"),$("target").value);
  rows=rows.filter(x=>x.target_vlm===$("target").value);
  fill($("task"),unique(rows,"task"),$("task").value);
  rows=rows.filter(x=>x.task===$("task").value);
  state.visible=rows;
  $("task").querySelectorAll("option").forEach(o=>o.textContent=formatTask(o.value));
  fill($("example"),rows.map(x=>x.title),$("example").value);
  state.selected=rows.find(x=>x.title===$("example").value)||rows[0];
  render();
}

function typeInto(node,text,speed=2){
  clearInterval(state.timer); node.textContent=""; node.classList.add("typing");
  let i=0,chunk=Math.max(1,Math.ceil(text.length/180));
  state.timer=setInterval(()=>{i=Math.min(text.length,i+chunk);node.textContent=text.slice(0,i);if(i>=text.length){clearInterval(state.timer);node.classList.remove("typing")}},speed*chunk);
}

function render(animate=false){
  const x=state.selected;if(!x)return;
  $("title").textContent=x.title;$("example-count").textContent=`${state.examples.indexOf(x)+1} of ${state.examples.length} recorded examples`;
  $("proposer-badge").textContent=`Proposer · ${x.proposer_model}`;$("target-badge").textContent=`Target · ${x.target_vlm}`;
  $("ultrasound").src=`public/${x.image}`;$("ultrasound").alt=`Ultrasound image for ${x.title}, record ${x.key}`;
  $("before").textContent=x.prediction_before;$("after").textContent=x.prediction_after;$("truth").textContent=x.ground_truth;
  $("changes").innerHTML=x.changes.map(c=>`<li><b>Step ${c.step}</b><br><del>${escapeHtml(c.previous)}</del> → <ins>${escapeHtml(c.replacement)}</ins></li>`).join("");
  $("record-key").textContent=x.key;
  $("dataset-source").textContent=`${x.dataset_source} · row ${x.dataset_row_index}`;
  if(animate){typeInto($("original-prompt"),x.original_prompt);setTimeout(()=>typeInto($("attacked-prompt"),x.attacked_prompt),650)}
  else{$("original-prompt").textContent=x.original_prompt;$("attacked-prompt").textContent=x.attacked_prompt}
}
function escapeHtml(value){const d=document.createElement("div");d.textContent=value;return d.innerHTML}

fetch("public/data/examples.json").then(r=>r.json()).then(data=>{
  state.examples=data;fill($("proposer"),unique(data,"proposer_model"));updateFilters("proposer");
  ["proposer","target","task","example"].forEach(id=>$(id).addEventListener("change",()=>updateFilters(id)));
  $("replay").addEventListener("click",()=>render(true));
}).catch(error=>{document.querySelector("main").innerHTML=`<p>Could not load examples: ${error.message}</p>`});
