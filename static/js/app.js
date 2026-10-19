(() => {
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];
  const store = {
    get(key, fallback) {
      try {
        const value = localStorage.getItem(`neurocare_${key}`);
        return value === null ? fallback : JSON.parse(value);
      } catch {
        return fallback;
      }
    },
    set(key, value) {
      try {
        localStorage.setItem(`neurocare_${key}`, JSON.stringify(value));
      } catch {}
    },
    remove(key) {
      try {
        localStorage.removeItem(`neurocare_${key}`);
      } catch {}
    }
  };

  const toast = (message) => {
    const el = $('#toast');
    if (!el) return;
    el.textContent = message;
    el.classList.add('show');
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => el.classList.remove('show'), 2800);
  };

  const titles = {
    dashboard: 'Dashboard',
    spiral: 'Spiral analysis',
    gait: 'Gait analysis',
    voice: 'Voice analysis',
    history: 'Assessment history',
    profile: 'My profile',
    settings: 'Settings',
    about: 'About NeuroCare'
  };

  let authMode = 'login';
  let currentPage = 'dashboard';
  const demoAccounts = {
    'demo-healthy': {
      id: 'demo-healthy',
      name: 'Healthy control demo',
      email: 'healthy.demo@neurocare.local',
      password: 'healthy123',
      label: 'Healthy control demo'
    },
    'demo-parkinson': {
      id: 'demo-parkinson',
      name: 'Parkinson participant demo',
      email: 'parkinson.demo@neurocare.local',
      password: 'parkinson123',
      label: 'Parkinson participant demo'
    }
  };
  const demoHistories = window.NEUROCARE_DEMO_HISTORIES || {};
  const modelAccuracies = window.NEUROCARE_MODEL_ACCURACIES || {};

  function currentAccountId() {
    const profile = store.get('profile', null);
    return profile?.id || profile?.email?.toLowerCase() || 'guest';
  }

  function historyKey() {
    return `history_${currentAccountId()}`;
  }

  function openAccount(account) {
    store.set('profile', {id: account.id, name: account.name, email: account.email, demo: true});
    const key = `history_${account.id}`;
    if (store.get(key, null) === null) store.set(key, demoHistories[account.id] || []);
    applyProfile();
    closeAuth();
    renderDashboard();
    if (currentPage === 'history') renderHistory();
    toast(`${account.label} opened. This is sample research data.`);
  }

  function navigate(page) {
    if (!titles[page]) page = 'dashboard';
    currentPage = page;

    $$('.page').forEach(el =>
      el.classList.toggle('active', el.id === `page-${page}`)
    );
    $$('.nav-item[data-page]').forEach(el =>
      el.classList.toggle('active', el.dataset.page === page)
    );

    $('#crumbCurrent').textContent = titles[page];
    $('#sidebar').classList.remove('open');

    if (location.hash !== `#${page}`) {
      history.replaceState(null, '', `#${page}`);
    }

    if (page === 'history') renderHistory();
    if (page === 'dashboard') renderDashboard();

    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  $$('[data-page]').forEach(el =>
    el.addEventListener('click', () => navigate(el.dataset.page))
  );

  $('#mobileMenu')?.addEventListener('click', () =>
    $('#sidebar').classList.toggle('open')
  );

  function openAuth(mode = 'login') {
    authMode = mode;
    $('#authView').hidden = false;
    $('#appShell').style.display = 'none';
    updateAuthMode();
    window.scrollTo(0, 0);
  }

  function closeAuth() {
    $('#authView').hidden = true;
    $('#appShell').style.display = '';
  }

  function updateAuthMode() {
    const signup = authMode === 'signup';

    $('#authEyebrow').textContent = signup
      ? 'CREATE YOUR SPACE'
      : 'WELCOME BACK';
    $('#authTitle').textContent = signup
      ? 'Create your account'
      : 'Sign in to your space';
    $('#authSubtitle').textContent = signup
      ? 'Set up a personal space for your assessment history.'
      : 'Your personal assessment history, all in one place.';
    $('#authSubmit').textContent = signup
      ? 'Create demo account'
      : 'Sign in';
    $('#signupExtra').hidden = !signup;
    $('#authName').required = signup;

    $('#authSwitch').innerHTML = signup
      ? 'Already have an account? <button class="text-btn" id="switchAuth" type="button">Sign in</button>'
      : 'New to NeuroCare? <button class="text-btn" id="switchAuth" type="button">Create an account</button>';

    $('#switchAuth').addEventListener('click', () => {
      authMode = signup ? 'login' : 'signup';
      updateAuthMode();
    });
  }

  $('#openLogin')?.addEventListener('click', () => openAuth('login'));
  $('#authBack')?.addEventListener('click', closeAuth);

  $('#authForm')?.addEventListener('submit', e => {
    e.preventDefault();

    const email = $('#authEmail').value.trim();
    const password = $('#authPassword').value;
    const demoAccount = Object.values(demoAccounts).find(
      account => account.email.toLowerCase() === email.toLowerCase()
    );
    if (demoAccount) {
      if (password !== demoAccount.password) {
        return toast('That sample account password does not match.');
      }
      openAccount(demoAccount);
      return;
    }
    const name =
      (authMode === 'signup' ? $('#authName').value.trim() : '') ||
      email.split('@')[0] ||
      'Guest';

    const profile = { id: email.toLowerCase(), name, email, demo: false };
    store.set('profile', profile);
    applyProfile();
    closeAuth();
    renderDashboard();

    toast(
      authMode === 'signup'
        ? 'Demo profile created on this device.'
        : 'Demo sign-in complete. Authentication is not connected.'
    );
  });

  $$('.demo-login').forEach(button => {
    button.addEventListener('click', () => {
      const account = demoAccounts[button.dataset.demoLogin];
      if (account) openAccount(account);
    });
  });

  $('#forgotBtn')?.addEventListener('click', () =>
    toast('Password recovery needs a connected authentication service.')
  );
  $('#profileShortcut')?.addEventListener('click', () => navigate('settings'));
  $('#avatarShortcut')?.addEventListener('click', () => navigate('profile'));
  $('#reminderBtn')?.addEventListener('click', () => navigate('settings'));
  $('#dashboardReminder')?.addEventListener('click', () => navigate('settings'));

  function applyProfile() {
    const profile = store.get('profile', null);
    const name = profile?.name || 'there';

    $('#welcomeName').textContent =
      name === 'there' ? ', there' : `, ${name.split(' ')[0]}`;
    $('#avatarShortcut').textContent =
      name === 'there' ? 'P' : name.charAt(0).toUpperCase();
    $('#profileAvatar').textContent =
      name === 'there' ? 'P' : name.charAt(0).toUpperCase();
    $('#profileDisplayName').textContent = profile?.name || 'Your name';
    $('#profileDisplayEmail').textContent =
      profile?.email || 'you@example.com';
    $('#profileName').value = profile?.name || '';
    $('#profileEmail').value = profile?.email || '';
    const notice = $('#demoNotice');
    if (notice) notice.hidden = !profile?.demo;
  }

  $('#profileForm')?.addEventListener('submit', e => {
    e.preventDefault();

    const name = $('#profileName').value.trim();
    const email = $('#profileEmail').value.trim();

    if (!name || !email) {
      return toast('Please enter your name and email.');
    }

    const existing = store.get('profile', null);
    store.set('profile', {...existing, id: existing?.id || email.toLowerCase(), name, email});
    applyProfile();
    toast('Profile updated in this browser.');
  });

  const historyItems = () => {
    const items = store.get(historyKey(), null);
    return Array.isArray(items) ? items : (currentAccountId() === 'guest' ? store.get('history', []) : []);
  };

  function sessionIdFor(item) {
    if (item.sessionId) return item.sessionId;
    const d = new Date(item.date);
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  }

  function summarizeSession(items) {
    const latestByType = new Map();
    items
      .filter(item => Number.isFinite(Number(item.probability)))
      .slice()
      .sort((a, b) => new Date(b.date) - new Date(a.date))
      .forEach(item => {
        if (!latestByType.has(item.type)) latestByType.set(item.type, item);
      });
    const components = [...latestByType.values()].map(item => ({
      type: item.type,
      score: Number(item.probability),
      accuracy: Number(item.accuracy ?? modelAccuracies[item.type] ?? 0)
    })).filter(item => item.accuracy > 0);
    const totalWeight = components.reduce((sum, item) => sum + item.accuracy, 0);
    if (!components.length || !totalWeight) return null;
    return {
      score: components.reduce((sum, item) => sum + item.score * item.accuracy, 0) / totalWeight,
      components
    };
  }

  function renderWeightedSummary(items) {
    const value = $('#weightedScoreValue');
    const detail = $('#weightedScoreDetail');
    if (!value || !detail) return;
    const sessions = new Map();
    items.forEach(item => {
      const id = sessionIdFor(item);
      if (!sessions.has(id)) sessions.set(id, []);
      sessions.get(id).push(item);
    });
    const latest = [...sessions.entries()].sort((a, b) => b[0].localeCompare(a[0]))[0];
    const summary = latest ? summarizeSession(latest[1]) : null;
    if (!summary) {
      value.textContent = 'No model results saved yet';
      detail.textContent = 'Save one or more model results on the same date to see the weighted research score.';
      return;
    }
    value.textContent = `${(summary.score * 100).toFixed(1)}% · ${summary.components.length} ${summary.components.length === 1 ? 'model' : 'models'}`;
    detail.textContent = summary.components
      .map(part => `${part.type.replace(' analysis', '')}: ${(part.score * 100).toFixed(1)}% score × ${(part.accuracy * 100).toFixed(1)}% validation accuracy`)
      .join(' · ');
  }

  function renderDashboard() {
    applyProfile();

    const items = historyItems();
    renderWeightedSummary(items);
    $('#assessmentCount').textContent = items.length;
    $('#lastAssessment').textContent = items.length
      ? new Date(items[0].date).toLocaleDateString(undefined, {
          month: 'short',
          day: '2-digit',
          year: 'numeric'
        })
      : 'Not yet';
    $('#lastAssessmentType').textContent = items.length
      ? items[0].type
      : 'Ready when you are';

    const recent = $('#recentList');

    if (!items.length) {
      recent.innerHTML =
        '<div class="empty-state"><span>▤</span><p>No assessments yet.</p><small>Your saved assessments will appear here.</small></div>';
    } else {
      recent.innerHTML = items
        .slice(0, 4)
        .map(
          item =>
            `<div class="recent-item"><span class="recent-icon">${iconFor(item.type)}</span><div class="recent-copy"><strong>${escapeHtml(item.type)}</strong><small>${formatDate(item.date)} · ${escapeHtml(item.result)}</small></div><span class="recent-arrow">›</span></div>`
        )
        .join('');
    }

    renderChart(items);

    const reminder = store.get('reminder', null);
    $('#nextReminder').textContent = reminder?.date
      ? new Date(`${reminder.date}T${reminder.time || '09:00'}`)
          .toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
      : 'Not set';
  }

  function renderChart(items) {
    const el = $('#activityChart');
    if (!el) return;

    if (!items.length) {
      el.innerHTML =
        '<div class="chart-empty"><span>↗</span><strong>Your progress takes shape over time.</strong><p>Complete an assessment to start building your personal history.</p></div>';
      return;
    }

    const counts = {};
    items
      .slice()
      .reverse()
      .forEach(item => {
        const d = new Date(item.date);
        const key = d.toLocaleDateString(undefined, { month: 'short' });
        counts[key] = (counts[key] || 0) + 1;
      });

    const labels = Object.keys(counts).slice(-6);
    const max = Math.max(...labels.map(k => counts[k]), 1);
    const points = labels
      .map(
        (key, i) =>
          `${(i / Math.max(labels.length - 1, 1)) * 92 + 4},${83 -
            (counts[key] / max) * 60}`
      )
      .join(' ');

    el.innerHTML = `<svg viewBox="0 0 100 100" preserveAspectRatio="none" style="width:100%;height:150px;overflow:visible" aria-label="Assessment activity chart"><line x1="4" y1="25" x2="96" y2="25" stroke="#e9ece3" stroke-dasharray="1 2"/><line x1="4" y1="55" x2="96" y2="55" stroke="#e9ece3" stroke-dasharray="1 2"/><line x1="4" y1="85" x2="96" y2="85" stroke="#e9ece3"/><polyline points="${points}" fill="none" stroke="#71866a" stroke-width="1.8" vector-effect="non-scaling-stroke" stroke-linecap="round" stroke-linejoin="round"/>${labels
      .map((k, i) => {
        const x = (i / Math.max(labels.length - 1, 1)) * 92 + 4;
        const y = 83 - (counts[k] / max) * 60;
        return `<circle cx="${x}" cy="${y}" r="1.5" fill="#71866a" vector-effect="non-scaling-stroke"/>`;
      })
      .join('')}</svg><div style="display:flex;justify-content:space-around;font-size:9px;color:#8c9487;margin-top:4px">${labels
      .map(k => `<span>${escapeHtml(k)}</span>`)
      .join('')}</div>`;
  }

  function renderHistory() {
    const items = historyItems();
    const q = ($('#historySearch')?.value || '').toLowerCase();
    const filter = $('#historyFilter')?.value || 'all';

    const visible = items.filter(
      i =>
        (filter === 'all' || i.type === filter) &&
        `${i.type} ${i.result} ${i.date} ${i.sourceNote || ''} ${i.sampleId || ''}`.toLowerCase().includes(q)
    );

    renderWeightedSummary(items);

    $('#historyRows').innerHTML = visible.length
      ? visible
          .map(
            i =>
              `<tr><td>${formatDate(i.date)}</td><td>${iconFor(i.type)} &nbsp; ${escapeHtml(i.type)}</td><td><span class="record-chip">${escapeHtml(i.result)}</span></td><td>${escapeHtml(i.sourceNote || (i.source === 'model' ? 'Saved model result' : 'Demo record'))}</td></tr>`
          )
          .join('')
      : '<tr><td colspan="4" class="table-empty">No matching records. Save a model result to start your history.</td></tr>';
  }

  $('#historySearch')?.addEventListener('input', renderHistory);
  $('#historyFilter')?.addEventListener('change', renderHistory);

  $('#exportHistory')?.addEventListener('click', () => {
    const items = historyItems();
    if (!items.length) return toast('There is no history to export yet.');

    const csv = [
      'Date,Session,Assessment,Result,PD-class score,Validation accuracy,Source,Sample note',
      ...items.map(i =>
        [i.date, i.sessionId || '', i.type, i.result, i.probability ?? '', i.accuracy ?? '', i.sourceNote || i.source || 'demo']
          .map(v => `"${String(v).replaceAll('"', '""')}"`)
          .join(',')
      )
    ].join('\n');

    const url = URL.createObjectURL(
      new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    );
    const a = document.createElement('a');
    a.href = url;
    a.download = 'neurocare-assessment-history.csv';
    a.click();
    URL.revokeObjectURL(url);
    toast('History CSV exported.');
  });

  function iconFor(type) {
    return type === 'Gait analysis'
      ? '♙'
      : type === 'Voice analysis'
        ? '♫'
        : '◎';
  }

  function formatDate(date) {
    return new Date(date).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, c => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[c]));
  }

  const input = $('#spiralInput');
  const dropzone = $('#dropzone');

  function showFile(file) {
    if (!file) return;

    if (
      !/^image\/(jpeg|png)$/.test(file.type) &&
      !/\.(jpe?g|png)$/i.test(file.name)
    ) {
      toast('Please choose a JPG or PNG image.');
      input.value = '';
      return;
    }

    if (file.size > 12 * 1024 * 1024) {
      toast('Please choose an image smaller than 12 MB.');
      input.value = '';
      return;
    }

    $('#filePreview').hidden = false;
    $('#fileName').textContent = file.name;
    $('#fileSize').textContent = `${(file.size / 1024).toFixed(0)} KB`;

    const reader = new FileReader();
    reader.onload = e => {
      $('#previewImage').src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  input?.addEventListener('change', () => showFile(input.files[0]));

  $('#removeFile')?.addEventListener('click', () => {
    input.value = '';
    $('#filePreview').hidden = true;
    $('#previewImage').removeAttribute('src');
  });

  ['dragenter', 'dragover'].forEach(evt =>
    dropzone?.addEventListener(evt, e => {
      e.preventDefault();
      dropzone.classList.add('dragging');
    })
  );

  ['dragleave', 'drop'].forEach(evt =>
    dropzone?.addEventListener(evt, e => {
      e.preventDefault();
      dropzone.classList.remove('dragging');
    })
  );

  dropzone?.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0];
    if (!file) return;

    try {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
    } catch {}

    showFile(file);
  });

  $('#spiralForm')?.addEventListener('submit', e => {
    if (!input.files.length) {
      e.preventDefault();
      toast('Choose a spiral image first.');
    } else {
      const button = $('.analyze-btn');
      button.disabled = true;
      button.querySelector('span').textContent = 'Analysing…';
    }
  });

  function saveModelResult(button) {
    const heading =
      $('.result-live h3')?.textContent.trim() || 'Model output available';
    const type = button.dataset.type || 'Spiral analysis';
    const probability = Number(button.dataset.probability);
    if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
      return toast('This result has no readable model score to save.');
    }
    const now = new Date();
    const sessionId = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    const item = {
      date: now.toISOString(),
      sessionId,
      type,
      probability,
      accuracy: Number(button.dataset.modelAccuracy || modelAccuracies[type] || 0),
      result: `${heading} · ${(probability * 100).toFixed(1)}% PD-class score`,
      source: 'model'
    };

    const items = historyItems();
    items.unshift(item);
    store.set(historyKey(), items);
    renderDashboard();
    renderHistory();
    toast(`${type} result saved. Same-date scores are combined with validation-accuracy weights.`);
  }

  $$('.save-model-result').forEach(button =>
    button.addEventListener('click', () => saveModelResult(button))
  );

  const voiceInput = $('#voiceInput');
  const recordButton = $('#recordVoice');
  const recordStatus = $('#recordStatus');
  let recorder;
  let microphone;
  let recordedChunks = [];

  const toWav = async blob => {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const context = new AudioCtx();
    const decoded = await context.decodeAudioData(await blob.arrayBuffer());
    const source = decoded.getChannelData(0);

    // Reject silence before converting it into a WAV file.
    let squareSum = 0;
    let inputPeak = 0;

    for (const sample of source) {
      squareSum += sample * sample;
      inputPeak = Math.max(inputPeak, Math.abs(sample));
    }

    const inputRms = Math.sqrt(squareSum / Math.max(source.length, 1));

    if (inputPeak < 0.003 || inputRms < 0.0006) {
      await context.close();
      throw new Error('quiet-recording');
    }

    const targetRate = 8000;
    const length = Math.max(
      1,
      Math.floor((source.length * targetRate) / decoded.sampleRate)
    );
    const samples = new Float32Array(length);

    for (let i = 0; i < length; i++) {
      const position = (i * decoded.sampleRate) / targetRate;
      const left = Math.floor(position);
      const fraction = position - left;

      samples[i] =
        (source[left] || 0) * (1 - fraction) +
        (source[Math.min(left + 1, source.length - 1)] || 0) * fraction;
    }

    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const write = (offset, text) => {
      [...text].forEach((ch, i) => view.setUint8(offset + i, ch.charCodeAt(0)));
    };

    write(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    write(8, 'WAVE');
    write(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, targetRate, true);
    view.setUint32(28, targetRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    write(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    for (let i = 0; i < samples.length; i++) {
      const x = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(44 + i * 2, x < 0 ? x * 32768 : x * 32767, true);
    }

    await context.close();
    return new File([buffer], 'voice-recording.wav', { type: 'audio/wav' });
  };

  recordButton?.addEventListener('click', async () => {
    if (recorder?.state === 'recording') {
      recorder.stop();
      recordButton.disabled = true;
      recordStatus.textContent = 'Preparing your recording…';
      return;
    }

    try {
      microphone = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });

      recordedChunks = [];
      recorder = new MediaRecorder(microphone);

      recorder.ondataavailable = event => {
        if (event.data.size) recordedChunks.push(event.data);
      };

      recorder.onstop = async () => {
        try {
          const wav = await toWav(
            new Blob(recordedChunks, {
              type: recorder.mimeType || 'audio/webm'
            })
          );

          const transfer = new DataTransfer();
          transfer.items.add(wav);
          voiceInput.files = transfer.files;
          recordStatus.textContent = 'Recording ready: voice-recording.wav';
        } catch (error) {
          // Clear any earlier file so it cannot be submitted by mistake.
          voiceInput.value = '';

          recordStatus.textContent =
            error?.message === 'quiet-recording'
              ? 'No clear voice was detected. Try again and say “ah” for about 4 seconds.'
              : 'Could not convert the recording. Please upload a WAV file.';
        }

        microphone?.getTracks().forEach(track => track.stop());
        microphone = null;
        recordButton.disabled = false;
        recordButton.textContent = '● Record again';
      };

      recorder.start();
      recordButton.textContent = '■ Stop recording';
      recordStatus.textContent = 'Say “ah” steadily for about 4 seconds, then stop.';

      setTimeout(() => {
        if (recorder?.state === 'recording') recorder.stop();
      }, 8000);
    } catch {
      recordStatus.textContent =
        'Microphone access was unavailable. Choose a WAV file instead.';
    }
  });

  $('#voiceForm')?.addEventListener('submit', e => {
    if (!voiceInput.files.length) {
      e.preventDefault();
      toast('Record or choose a WAV file first.');
    } else {
      const button = $('#voiceForm .analyze-btn');
      button.disabled = true;
      button.querySelector('span').textContent = 'Analysing…';
    }
  });

  function loadReminder() {
    const r = store.get('reminder', null);
    if (r) {
      $('#reminderDate').value = r.date || '';
      $('#reminderTime').value = r.time || '';
      $('#reminderStatus').textContent =
        `Reminder saved for ${r.date}${r.time ? ` at ${r.time}` : ''}. This demo cannot send notifications while closed.`;
    }
  }

  $('#saveReminder')?.addEventListener('click', () => {
    const date = $('#reminderDate').value;
    const time = $('#reminderTime').value;

    if (!date) return toast('Choose a reminder date first.');

    store.set('reminder', { date, time });
    loadReminder();
    renderDashboard();
    toast('Reminder saved in this browser.');
  });

  $('#clearReminder')?.addEventListener('click', () => {
    store.remove('reminder');
    $('#reminderDate').value = '';
    $('#reminderTime').value = '';
    $('#reminderStatus').textContent =
      'No reminder set. This demo cannot send notifications while closed.';
    renderDashboard();
    toast('Reminder cleared.');
  });

  $('#clearLocalData')?.addEventListener('click', () => {
    if (
      confirm(
        'Clear this browser’s NeuroCare demo profile, history and reminder? This cannot be undone.'
      )
    ) {
      [historyKey(), 'profile', 'history', 'reminder'].forEach(store.remove);
      applyProfile();
      renderDashboard();
      renderHistory();
      loadReminder();
      toast('Local demo data cleared.');
    }
  });

  // Saving a result is user-triggered; nothing is added to history automatically.
  const resultHeading = $('.result-live h3');
  if (resultHeading) {
    const resultText = resultHeading.textContent.trim();
    const existing = historyItems();

    if (
      !existing.some(
        i =>
          i.source === 'model' &&
          i.result.startsWith(resultText) &&
          Date.now() - new Date(i.date).getTime() < 60000
      )
    ) {
      // The result remains unsaved until the user clicks its save button.
    }
  }

  applyProfile();
  renderDashboard();
  loadReminder();

  const initialPage =
    location.hash.replace('#', '') || document.body.dataset.initialPage;
  if (titles[initialPage]) navigate(initialPage);
})();
