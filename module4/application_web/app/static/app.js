const state = { models: [], active: null };
const dialog = document.querySelector('#lab-dialog');
const form = document.querySelector('#prediction-form');
const resultPanel = document.querySelector('#result-panel');
const exampleButton = document.querySelector('#load-example');

const labDescriptions = {
  g1: 'Testez le modèle avec les dix-sept caractéristiques réellement utilisées dans le notebook.',
  g2: 'Identifiez la variété la plus proche à partir de seize mesures morphologiques.',
  g3: 'Prévoyez la température maximale du lendemain à partir des conditions de la veille.',
  g4: 'Examinez une classification clinique dans un cadre strictement méthodologique.',
  g5: 'Estimez la valeur associée à un ensemble d’indicateurs nationaux.',
  g6: 'Comparez un passage aux quatre ouvrages connus du modèle.'
};

function metricValue(value) {
  if (typeof value !== 'number') return value;
  if (value > 0 && value <= 1) return `${(value * 100).toFixed(1)} %`;
  return value.toFixed(2);
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Le calcul n’a pas abouti.');
  return data;
}

function renderCards() {
  const grid = document.querySelector('#lab-grid');
  grid.innerHTML = state.models.map((model, index) => `
    <button class="lab-card" data-open="${model.id}">
      <span class="lab-index"><span>TP ${index + 1}</span><b>${model.method.split(' ')[0].toUpperCase()}</b></span>
      <span><h3>${model.title}</h3><p>${labDescriptions[model.id]}</p></span>
      <span class="lab-open">Accéder au laboratoire →</span>
    </button>`).join('');
}

function renderMetrics(metrics) {
  return Object.entries(metrics).map(([label, value]) => `
    <div class="metric-row"><span>${label}</span><strong>${metricValue(value)}</strong></div>`).join('');
}

function numberFields(model) {
  return `<div class="form-grid">${model.fields.map(field => {
    if (field.type === 'choice') {
      return `<div class="field"><label for="${field.name}">${field.label}</label>
        <select id="${field.name}" name="${field.name}">${field.options.map(option => `<option value="${option.value}">${option.label}</option>`).join('')}</select></div>`;
    }
    if (field.type === 'boolean') {
      return `<div class="field"><label for="${field.name}">${field.label}</label>
        <select id="${field.name}" name="${field.name}"><option value="0">Non</option><option value="1">Oui</option></select></div>`;
    }
    return `<div class="field"><label for="${field.name}">${field.label}</label>
      <input id="${field.name}" name="${field.name}" type="number" step="${field.step}" required>
      <small>Plage centrale : ${field.min} à ${field.max}</small></div>`;
  }).join('')}</div>`;
}

function renderForm(model) {
  exampleButton.hidden = false;
  document.querySelector('#form-title').textContent = 'Renseignez les données à analyser';
  if (model.id === 'g6') {
    document.querySelector('#form-title').textContent = 'Collez un passage en anglais';
    form.innerHTML = `<div class="field"><label for="text">Passage à examiner</label><textarea id="text" name="text" placeholder="Collez ici un passage d’au moins 25 mots…" required></textarea><small>Le modèle compare uniquement avec Equiano, Plaatje, Conrad et Kingsley.</small></div>`;
  } else {
    form.innerHTML = numberFields(model);
  }
}

function fillExample() {
  const model = state.active;
  if (!model) return;
  if (model.id === 'g6') {
    form.elements.text.value = model.example;
    return;
  }
  let example = model.example;
  if (model.id === 'g2') example = model.examples[Object.keys(model.examples)[0]];
  if (!example) example = Object.fromEntries(model.fields.map(field => [field.name, field.default]));
  Object.entries(example).forEach(([name, value]) => {
    if (form.elements[name]) form.elements[name].value = value;
  });
}

function openLab(id) {
  const model = state.models.find(item => item.id === id);
  if (!model) return;
  state.active = model;
  document.querySelector('#dialog-kicker').textContent = `Travail pratique ${id.slice(1)}`;
  document.querySelector('#dialog-title').textContent = model.title;
  document.querySelector('#dialog-question').textContent = model.question;
  document.querySelector('#dialog-method').textContent = model.method;
  document.querySelector('#dialog-scope').textContent = model.scope;
  document.querySelector('#dialog-warning').textContent = model.warning;
  document.querySelector('#dialog-metrics').innerHTML = renderMetrics(model.metrics);
  renderForm(model);
  fillExample();
  resultPanel.className = 'result empty';
  resultPanel.innerHTML = '<p>Avant de lancer l’analyse, notez le résultat auquel vous vous attendez. Vous pourrez ensuite le confronter à celui du modèle.</p>';
  dialog.showModal();
  dialog.scrollTop = 0;
}

function warningHtml(warnings = []) {
  return warnings.length ? `<div class="result-warning">${warnings.join('<br>')}</div>` : '';
}

function probabilityRows(rows) {
  return `<div class="probabilities">${rows.slice(0, 7).map(row => `
    <div class="prob-row"><span>${row.label}</span><span class="prob-track"><i style="width:${Math.max(2, row.value * 100)}%"></i></span><strong>${(row.value * 100).toFixed(1)} %</strong></div>`).join('')}</div>`;
}

function showResult(model, data) {
  let html = '';
  if (model.id === 'g1') {
    html = `<span class="result-kicker">Classe prédite au seuil de 50 %</span><div class="result-value">${data.prediction}</div>
      <p class="result-note">Probabilité calculée de manque de soins : <strong>${(data.probability * 100).toFixed(1)} %</strong>. Comparez-la au seuil de décision de ${(data.threshold * 100).toFixed(0)} % et examinez les limites avant toute interprétation.</p>${warningHtml(data.warnings)}`;
  } else if (model.id === 'g2') {
    html = `<span class="result-kicker">Variété la plus proche</span><div class="result-value">${data.prediction}</div>
      <p class="result-note">Examinez aussi les probabilités attribuées aux sept variétés connues :</p>${probabilityRows(data.probabilities)}${warningHtml(data.warnings)}`;
  } else if (model.id === 'g3' || model.id === 'g5') {
    html = `<span class="result-kicker">Valeur prévue</span><div class="result-value">${data.prediction.toFixed(1)} ${data.unit}</div>
      <p class="result-note">Pour interpréter cette valeur, utilisez la zone indicative fondée sur 90 % des erreurs historiques : <strong>${data.interval[0].toFixed(1)} à ${data.interval[1].toFixed(1)} ${data.unit}</strong>.</p>${warningHtml(data.warnings)}`;
  } else if (model.id === 'g4') {
    html = `<span class="result-kicker">Similarité calculée</span><div class="result-value">${data.prediction}</div>
      <p class="result-note">Le modèle attribue <strong>${(data.probability * 100).toFixed(1)} %</strong> à la classe MRC. Interprétez cette valeur comme le comportement du modèle sur ce jeu de données, jamais comme un diagnostic.</p>${warningHtml(data.warnings)}`;
  } else if (model.id === 'g6') {
    html = `<span class="result-kicker">Ouvrage le plus proche</span><div class="result-value">${data.prediction}</div>
      <p class="result-note">Vous avez soumis ${data.word_count} mots. Examinez leur proximité lexicale avec chacun des quatre ouvrages.</p>${probabilityRows(data.probabilities)}
      <div class="chips">${data.indicators.map(word => `<span class="chip">${word}</span>`).join('')}</div>${warningHtml(data.warnings)}`;
  }
  resultPanel.className = 'result';
  resultPanel.innerHTML = html;
  resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function submit(event) {
  event.preventDefault();
  const model = state.active;
  const formState = document.querySelector('#form-state');
  formState.textContent = 'Analyse en cours…';
  try {
    let endpoint, body;
    if (model.id === 'g6') {
      endpoint = '/api/g6/predict';
      body = { text: form.elements.text.value };
    } else {
      endpoint = `/api/${model.id}/predict`;
      body = { values: Object.fromEntries(model.fields.map(field => [field.name, Number(form.elements[field.name].value)])) };
    }
    const data = await request(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    showResult(model, data);
    formState.textContent = 'Analyse terminée.';
  } catch (error) {
    resultPanel.className = 'result';
    resultPanel.innerHTML = `<span class="result-kicker">Vérifiez les données saisies</span><p class="result-warning">${error.message}</p>`;
    formState.textContent = 'L’analyse n’a pas abouti.';
  }
}

async function init() {
  state.models = await request('/api/models');
  renderCards();
  document.addEventListener('click', event => {
    const target = event.target.closest('[data-open]');
    if (target) openLab(target.dataset.open);
  });
  document.querySelector('#close-dialog').addEventListener('click', () => dialog.close());
  exampleButton.addEventListener('click', fillExample);
  form.addEventListener('submit', submit);
  const requestedLab = new URLSearchParams(window.location.search).get('lab');
  if (requestedLab && state.models.some(model => model.id === requestedLab)) openLab(requestedLab);
}

init().catch(error => {
  document.querySelector('#lab-grid').innerHTML = `<p>Impossible de charger les modèles : ${error.message}</p>`;
});
