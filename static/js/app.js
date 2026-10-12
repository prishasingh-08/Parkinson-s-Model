(() => {
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];
  const store = {
    get(key, fallback) { try { const value = localStorage.getItem(`neurocare_${key}`); return value === null ? fallback : JSON.parse(value); } catch { return fallback; } },
    set(key, value) { try { localStorage.setItem(`neurocare_${key}`, JSON.stringify(value)); } catch {} },
    remove(key) { try { localStorage.removeItem(`neurocare_${key}`); } catch {} }
  };
  const toast = (message) => { const el = $('#toast'); if (!el) return; el.textContent = message; el.classList.add('show'); clearTimeout(toast.timer); toast.timer = setTimeout(() => el.classList.remove('show'), 2800); };
  const titles = {dashboard:'Dashboard',spiral:'Spiral analysis',gait:'Gait analysis',voice:'Voice analysis',history:'Assessment history',profile:'My profile',settings:'Settings',about:'About NeuroCare'};
  let authMode = 'login';
  let currentPage = 'dashboard';

  function navigate(page) {
    if (!titles[page]) page = 'dashboard';
    currentPage = page;
    $$('.page').forEach(el => el.classList.toggle('active', el.id === `page-${page}`));
    $$('.nav-item[data-page]').forEach(el => el.classList.toggle('active', el.dataset.page === page));
    $('#crumbCurrent').textContent = titles[page];
    $('#sidebar').classList.remove('open');
    if (location.hash !== `#${page}`) history.replaceState(null, '', `#${page}`);
    if (page === 'history') renderHistory();
    if (page === 'dashboard') renderDashboard();
    window.scrollTo({top: 0, behavior: 'smooth'});
  }
  $$('[data-page]').forEach(el => el.addEventListener('click', () => navigate(el.dataset.page)));
  $('#mobileMenu')?.addEventListener('click', () => $('#sidebar').classList.toggle('open'));

  function openAuth(mode = 'login') {
    authMode = mode;
    $('#authView').hidden = false;
    $('#appShell').style.display = 'none';
    updateAuthMode();
    window.scrollTo(0, 0);
  }
  function closeAuth() { $('#authView').hidden = true; $('#appShell').style.display = ''; }
  function updateAuthMode() {
    const signup = authMode === 'signup';
    $('#authEyebrow').textContent = signup ? 'CREATE YOUR SPACE' : 'WELCOME BACK';
    $('#authTitle').textContent = signup ? 'Create your account' : 'Sign in to your space';
    $('#authSubtitle').textContent = signup ? 'Set up a personal space for your assessment history.' : 'Your personal assessment history, all in one place.';
    $('#authSubmit').textContent = signup ? 'Create demo account' : 'Sign in';
    $('#signupExtra').hidden = !signup;
    $('#authName').required = signup;
    $('#authSwitch').innerHTML = signup ? 'Already have an account? <button class="text-btn" id="switchAuth" type="button">Sign in</button>' : 'New to NeuroCare? <button class="text-btn" id="switchAuth" type="button">Create an account</button>';
    $('#switchAuth').addEventListener('click', () => { authMode = signup ? 'login' : 'signup'; updateAuthMode(); });
  }
  $('#openLogin')?.addEventListener('click', () => openAuth('login'));
  $('#authBack')?.addEventListener('click', closeAuth);
  $('#authForm')?.addEventListener('submit', e => {
    e.preventDefault();
    const email = $('#authEmail').value.trim();
    const name = (authMode === 'signup' ? $('#authName').value.trim() : '') || email.split('@')[0] || 'Guest';
    const profile = {name, email};
    store.set('profile', profile);
    applyProfile(); closeAuth(); renderDashboard();
    toast(authMode === 'signup' ? 'Demo profile created on this device.' : 'Demo sign-in complete. Authentication is not connected.');
  });
  $('#forgotBtn')?.addEventListener('click', () => toast('Password recovery needs a connected authentication service.'));
  $('#profileShortcut')?.addEventListener('click', () => navigate('settings'));
  $('#avatarShortcut')?.addEventListener('click', () => navigate('profile'));
  $('#reminderBtn')?.addEventListener('click', () => navigate('settings'));
  $('#dashboardReminder')?.addEventListener('click', () => navigate('settings'));

  function applyProfile() {
    const profile = store.get('profile', null);
    const name = profile?.name || 'there';
    $('#welcomeName').textContent = name === 'there' ? ', there' : `, ${name.split(' ')[0]}`;
    $('#avatarShortcut').textContent = name === 'there' ? 'P' : name.charAt(0).toUpperCase();
    $('#profileAvatar').textContent = name === 'there' ? 'P' : name.charAt(0).toUpperCase();
    $('#profileDisplayName').textContent = profile?.name || 'Your name';
    $('#profileDisplayEmail').textContent = profile?.email || 'you@example.com';
    $('#profileName').value = profile?.name || '';
    $('#profileEmail').value = profile?.email || '';
  }
  $('#profileForm')?.addEventListener('submit', e => {
    e.preventDefault();
    const name = $('#profileName').value.trim(); const email = $('#profileEmail').value.trim();
    if (!name || !email) return toast('Please enter your name and email.');
    store.set('profile', {name, email}); applyProfile(); toast('Profile updated in this browser.');
  });

  const historyItems = () => store.get('history', []);
  function renderDashboard() {
    applyProfile();
    const items = historyItems();
    $('#assessmentCount').textContent = items.length;
    $('#lastAssessment').textContent = items.length ? new Date(items[0].date).toLocaleDateString(undefined, {month:'short', day:'2-digit', year:'numeric'}) : 'Not yet';
    $('#lastAssessmentType').textContent = items.length ? items[0].type : 'Ready when you are';
    const recent = $('#recentList');
    if (!items.length) recent.innerHTML = '<div class="empty-state"><span>▤</span><p>No assessments yet.</p><small>Your saved assessments will appear here.</small></div>';
    else recent.innerHTML = items.slice(0, 4).map(item => `<div class="recent-item"><span class="recent-icon">${iconFor(item.type)}</span><div class="recent-copy"><strong>${escapeHtml(item.type)}</strong><small>${formatDate(item.date)} · ${escapeHtml(item.result)}</small></div><span class="recent-arrow">›</span></div>`).join('');
    renderChart(items);
    const reminder = store.get('reminder', null);
    $('#nextReminder').textContent = reminder?.date ? new Date(`${reminder.date}T${reminder.time || '09:00'}`).toLocaleDateString(undefined, {month:'short',day:'numeric'}) : 'Not set';
  }
  function renderChart(items) {
    const el = $('#activityChart'); if (!el) return;
    if (!items.length) { el.innerHTML = '<div class="chart-empty"><span>↗</span><strong>Your progress takes shape over time.</strong><p>Complete an assessment to start building your personal history.</p></div>'; return; }
    const counts = {};
    items.slice().reverse().forEach(item => { const d = new Date(item.date); const key = d.toLocaleDateString(undefined,{month:'short'}); counts[key] = (counts[key] || 0) + 1; });
    const labels = Object.keys(counts).slice(-6); const max = Math.max(...labels.map(k=>counts[k]), 1);
    const points = labels.map((key,i) => `${(i/(Math.max(labels.length-1,1)))*92+4},${83-(counts[key]/max)*60}`).join(' ');
    el.innerHTML = `<svg viewBox="0 0 100 100" preserveAspectRatio="none" style="width:100%;height:150px;overflow:visible" aria-label="Assessment activity chart"><line x1="4" y1="25" x2="96" y2="25" stroke="#e9ece3" stroke-dasharray="1 2"/><line x1="4" y1="55" x2="96" y2="55" stroke="#e9ece3" stroke-dasharray="1 2"/><line x1="4" y1="85" x2="96" y2="85" stroke="#e9ece3"/><polyline points="${points}" fill="none" stroke="#71866a" stroke-width="1.8" vector-effect="non-scaling-stroke" stroke-linecap="round" stroke-linejoin="round"/>${labels.map((k,i)=>{const x=(i/(Math.max(labels.length-1,1)))*92+4,y=83-(counts[k]/max)*60;return `<circle cx="${x}" cy="${y}" r="1.5" fill="#71866a" vector-effect="non-scaling-stroke"/>`;}).join('')}</svg><div style="display:flex;justify-content:space-around;font-size:9px;color:#8c9487;margin-top:4px">${labels.map(k=>`<span>${escapeHtml(k)}</span>`).join('')}</div>`;
  }
  function renderHistory() {
    const items = historyItems(); const q = ($('#historySearch')?.value || '').toLowerCase(); const filter = $('#historyFilter')?.value || 'all';
    const visible = items.filter(i => (filter === 'all' || i.type === filter) && `${i.type} ${i.result} ${i.date}`.toLowerCase().includes(q));
    $('#historyRows').innerHTML = visible.length ? visible.map(i=>`<tr><td>${formatDate(i.date)}</td><td>${iconFor(i.type)} &nbsp; ${escapeHtml(i.type)}</td><td><span class="record-chip">${escapeHtml(i.result)}</span></td><td>${i.source === 'model' ? 'Model result' : 'Demo record'}</td></tr>`).join('') : '<tr><td colspan="4" class="table-empty">No matching records. Save a model result to start your history.</td></tr>';
  }
  $('#historySearch')?.addEventListener('input', renderHistory);
  $('#historyFilter')?.addEventListener('change', renderHistory);
  $('#exportHistory')?.addEventListener('click', () => {
    const items = historyItems(); if (!items.length) return toast('There is no history to export yet.');
    const csv = ['Date,Assessment,Result,Source', ...items.map(i=>[i.date,i.type,i.result,i.source || 'demo'].map(v=>`"${String(v).replaceAll('"','""')}"`).join(','))].join('\n');
    const url = URL.createObjectURL(new Blob([csv], {type:'text/csv;charset=utf-8;'})); const a = document.createElement('a'); a.href=url; a.download='neurocare-assessment-history.csv'; a.click(); URL.revokeObjectURL(url); toast('History CSV exported.');
  });
  function iconFor(type) { return type === 'Gait analysis' ? '♙' : type === 'Voice analysis' ? '♫' : '◎'; }
  function formatDate(date) { return new Date(date).toLocaleDateString(undefined,{month:'short',day:'numeric',year:'numeric'}); }
  function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

  const input = $('#spiralInput'); const dropzone = $('#dropzone');
  function showFile(file) {
    if (!file) return;
    if (!/^image\/(jpeg|png)$/.test(file.type) && !/\.(jpe?g|png)$/i.test(file.name)) { toast('Please choose a JPG or PNG image.'); input.value=''; return; }
    if (file.size > 12 * 1024 * 1024) { toast('Please choose an image smaller than 12 MB.'); input.value=''; return; }
    $('#filePreview').hidden = false; $('#fileName').textContent = file.name; $('#fileSize').textContent = `${(file.size/1024).toFixed(0)} KB`;
    const reader = new FileReader(); reader.onload = e => { $('#previewImage').src = e.target.result; }; reader.readAsDataURL(file);
  }
  input?.addEventListener('change', () => showFile(input.files[0]));
  $('#removeFile')?.addEventListener('click', () => { input.value=''; $('#filePreview').hidden=true; $('#previewImage').removeAttribute('src'); });
  ['dragenter','dragover'].forEach(evt => dropzone?.addEventListener(evt, e => {e.preventDefault();dropzone.classList.add('dragging');}));
  ['dragleave','drop'].forEach(evt => dropzone?.addEventListener(evt, e => {e.preventDefault();dropzone.classList.remove('dragging');}));
  dropzone?.addEventListener('drop', e => {const file=e.dataTransfer.files[0];if(!file)return;try{const dt=new DataTransfer();dt.items.add(file);input.files=dt.files;}catch{}showFile(file);});
  $('#spiralForm')?.addEventListener('submit', e => { if (!input.files.length) { e.preventDefault(); toast('Choose a spiral image first.'); } else { const button=$('.analyze-btn'); button.disabled=true; button.querySelector('span').textContent='Analysing…'; } });

  $('#saveResult')?.addEventListener('click', () => {
    const heading = $('.result-live h3')?.textContent.trim() || 'Model output available';
    const prob = $('.result-prob')?.textContent.trim();
    const item = {date:new Date().toISOString(),type:'Spiral analysis',result:prob ? `${heading} (${prob})` : heading,source:'model'};
    const items = historyItems(); items.unshift(item); store.set('history', items); renderDashboard(); renderHistory(); toast('Spiral result saved to this browser.');
  });

  function loadReminder() { const r=store.get('reminder',null); if(r){$('#reminderDate').value=r.date||'';$('#reminderTime').value=r.time||'';$('#reminderStatus').textContent=`Reminder saved for ${r.date}${r.time?` at ${r.time}`:''}. This demo cannot send notifications while closed.`;} }
  $('#saveReminder')?.addEventListener('click', () => {const date=$('#reminderDate').value,time=$('#reminderTime').value;if(!date)return toast('Choose a reminder date first.');store.set('reminder',{date,time});loadReminder();renderDashboard();toast('Reminder saved in this browser.');});
  $('#clearReminder')?.addEventListener('click', () => {store.remove('reminder');$('#reminderDate').value='';$('#reminderTime').value='';$('#reminderStatus').textContent='No reminder set. This demo cannot send notifications while closed.';renderDashboard();toast('Reminder cleared.');});
  $('#clearLocalData')?.addEventListener('click', () => {if(confirm('Clear this browser’s NeuroCare demo profile, history and reminder? This cannot be undone.')){['profile','history','reminder'].forEach(store.remove);applyProfile();renderDashboard();renderHistory();loadReminder();toast('Local demo data cleared.');}});

  // If Flask rendered a result, offer a one-click save to this browser's demo history.
  const resultHeading = $('.result-live h3');
  if (resultHeading) {
    const resultText = resultHeading.textContent.trim();
    const existing = historyItems();
    if (!existing.some(i => i.source==='model' && i.result.startsWith(resultText) && Date.now()-new Date(i.date).getTime()<60000)) {
      // Save is intentionally user-triggered; no data is persisted until the user clicks the button.
    }
  }
  applyProfile(); renderDashboard(); loadReminder();
  const initialPage = location.hash.replace('#',''); if (titles[initialPage]) navigate(initialPage);
})();
