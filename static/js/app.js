/**
 * CapCut-Lite — Главный модуль приложения (app.js)
 * 
 * Отвечает за:
 * - Глобальное состояние (маркеры, выбранный маркер, видео)
 * - Загрузку видео
 * - Управление воспроизведением
 * - Синхронизацию UI (properties panel) с состоянием
 * - Сохранение/загрузку проектов
 * - Экспорт
 * - Уведомления
 * 
 * Зависит от: timeline.js, tts.js, audio.js (загружаются перед app.js)
 */

// ═══════════════════════════════════════════════════════════════════════
// ГЛОБАЛЬНОЕ СОСТОЯНИЕ
// ═══════════════════════════════════════════════════════════════════════

const AppState = {
    // Видео
    videoFile: null,       // имя загруженного файла
    videoUrl: null,        // URL для <video>
    videoDuration: 0,      // длительность в секундах

    // Маркеры (реплики на таймлайне)
    markers: [],           // [{id, text, startTime, duration, audioUrl, audioFile, language}]
    selectedMarkerId: null,// ID выбранного маркера

    // Состояние воспроизведения
    isPlaying: false,

    // Утилита: генерация ID
    nextId: 1,
    generateId() {
        return `marker_${this.nextId++}`;
    }
};

// ═══════════════════════════════════════════════════════════════════════
// ИНИЦИАЛИЗАЦИЯ
// ═══════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initVideoLoader();
    initPlayerControls();
    initToolbarActions();
    initPropertiesPanel();
    initMarkerActions();
    console.log('CapCut-Lite инициализирован ✓');
});

// ═══════════════════════════════════════════════════════════════════════
// ЗАГРУЗКА ВИДЕО
// ═══════════════════════════════════════════════════════════════════════

function initVideoLoader() {
    const videoInput = document.getElementById('video-input');
    const btnLoadVideo = document.getElementById('btn-load-video');
    const placeholder = document.getElementById('video-placeholder');
    const videoPlayer = document.getElementById('video-player');
    const videoContainer = document.querySelector('.video-container');

    // Кнопка "Загрузить видео"
    btnLoadVideo.addEventListener('click', () => videoInput.click());

    // Выбор файла
    videoInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        await loadVideo(file);
    });

    // Drag & Drop на контейнер видео
    videoContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        videoContainer.style.outline = '2px dashed var(--accent)';
    });

    videoContainer.addEventListener('dragleave', () => {
        videoContainer.style.outline = '';
    });

    videoContainer.addEventListener('drop', async (e) => {
        e.preventDefault();
        videoContainer.style.outline = '';
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('video/')) {
            await loadVideo(file);
        }
    });
}

async function loadVideo(file) {
    const placeholder = document.getElementById('video-placeholder');
    const videoPlayer = document.getElementById('video-player');

    notify('info', `Загрузка видео: ${file.name}...`);

    // Загружаем на сервер
    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await resp.json();

        if (!data.ok) throw new Error('Ошибка загрузки');

        AppState.videoFile = data.filename;
        AppState.videoUrl = data.url;

        // Показываем видеоплеер
        videoPlayer.src = data.url;
        videoPlayer.hidden = false;
        placeholder.hidden = true;

        // Ждём метаданные видео
        videoPlayer.addEventListener('loadedmetadata', () => {
            AppState.videoDuration = videoPlayer.duration;
            document.getElementById('btn-play').disabled = false;
            document.getElementById('btn-add-marker').disabled = false;
            updateTimeDisplay();
            Timeline.init(videoPlayer.duration);
            notify('success', `Видео загружено: ${formatTime(videoPlayer.duration)}`);
        }, { once: true });

    } catch (err) {
        notify('error', `Ошибка: ${err.message}`);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// УПРАВЛЕНИЕ ВОСПРОИЗВЕДЕНИЕМ
// ═══════════════════════════════════════════════════════════════════════

function initPlayerControls() {
    const videoPlayer = document.getElementById('video-player');
    const btnPlay = document.getElementById('btn-play');
    const volumeSlider = document.getElementById('volume-slider');

    btnPlay.addEventListener('click', togglePlayback);

    // Горячие клавиши
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
        if (e.code === 'Space') {
            e.preventDefault();
            togglePlayback();
        }
    });

    // Громкость
    volumeSlider.addEventListener('input', (e) => {
        videoPlayer.volume = e.target.value;
    });

    // Обновление времени при воспроизведении
    videoPlayer.addEventListener('timeupdate', () => {
        updateTimeDisplay();
        Timeline.updatePlayhead(videoPlayer.currentTime);
        AudioEngine.syncWithVideo(videoPlayer.currentTime);
    });

    videoPlayer.addEventListener('ended', () => {
        AppState.isPlaying = false;
        btnPlay.textContent = '▶ Играть';
        AudioEngine.stopAll();
    });
}

function togglePlayback() {
    const videoPlayer = document.getElementById('video-player');
    const btnPlay = document.getElementById('btn-play');

    if (!videoPlayer.src) return;

    if (AppState.isPlaying) {
        videoPlayer.pause();
        AudioEngine.pauseAll();
        btnPlay.textContent = '▶ Играть';
    } else {
        videoPlayer.play();
        AudioEngine.resumeAll(videoPlayer.currentTime);
        btnPlay.textContent = '⏸ Пауза';
    }
    AppState.isPlaying = !AppState.isPlaying;
}

function seekTo(time) {
    const videoPlayer = document.getElementById('video-player');
    videoPlayer.currentTime = time;
    updateTimeDisplay();
    Timeline.updatePlayhead(time);
    AudioEngine.syncWithVideo(time);
}

function updateTimeDisplay() {
    const videoPlayer = document.getElementById('video-player');
    const display = document.getElementById('time-display');
    display.textContent = `${formatTime(videoPlayer.currentTime)} / ${formatTime(videoPlayer.duration || 0)}`;
}

// ═══════════════════════════════════════════════════════════════════════
// МАРКЕРЫ — CRUD
// ═══════════════════════════════════════════════════════════════════════

function initMarkerActions() {
    const btnAdd = document.getElementById('btn-add-marker');
    btnAdd.addEventListener('click', () => {
        const videoPlayer = document.getElementById('video-player');
        addMarker(videoPlayer.currentTime);
    });
}

function addMarker(startTime = 0, text = '', duration = 2) {
    const marker = {
        id: AppState.generateId(),
        text: text || 'Новая реплика',
        startTime: startTime,
        duration: duration,
        audioUrl: null,
        audioFile: null,
        language: 'ru-RU'
    };

    AppState.markers.push(marker);
    Timeline.renderMarker(marker);
    selectMarker(marker.id);
    notify('info', 'Маркер добавлен');
    return marker;
}

function deleteMarker(id) {
    AppState.markers = AppState.markers.filter(m => m.id !== id);
    Timeline.removeMarker(id);
    AudioEngine.removeAudio(id);

    if (AppState.selectedMarkerId === id) {
        AppState.selectedMarkerId = null;
        updatePropertiesPanel();
    }
    notify('info', 'Маркер удалён');
}

function selectMarker(id) {
    AppState.selectedMarkerId = id;
    Timeline.highlightMarker(id);
    updatePropertiesPanel();
}

function getSelectedMarker() {
    return AppState.markers.find(m => m.id === AppState.selectedMarkerId) || null;
}

function updateMarker(id, updates) {
    const marker = AppState.markers.find(m => m.id === id);
    if (!marker) return;
    Object.assign(marker, updates);
    Timeline.updateMarkerElement(marker);
    if (id === AppState.selectedMarkerId) {
        updatePropertiesPanel();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ПАНЕЛЬ СВОЙСТВ (Properties Panel)
// ═══════════════════════════════════════════════════════════════════════

function initPropertiesPanel() {
    const markerText = document.getElementById('marker-text');
    const markerStart = document.getElementById('marker-start');
    const markerDuration = document.getElementById('marker-duration');
    const markerLanguage = document.getElementById('marker-language');
    const btnDeleteMarker = document.getElementById('btn-delete-marker');
    const btnGenerateTts = document.getElementById('btn-generate-tts');
    const markerAudioInput = document.getElementById('marker-audio-input');
    const btnPlayAttached = document.getElementById('btn-play-attached');
    const btnRemoveAudio = document.getElementById('btn-remove-audio');

    // Обновление маркера при изменении полей
    markerText.addEventListener('input', () => {
        const marker = getSelectedMarker();
        if (marker) {
            marker.text = markerText.value;
            Timeline.updateMarkerElement(marker);
        }
    });

    markerStart.addEventListener('change', () => {
        const marker = getSelectedMarker();
        if (marker) {
            marker.startTime = parseFloat(markerStart.value) || 0;
            Timeline.updateMarkerElement(marker);
        }
    });

    markerDuration.addEventListener('change', () => {
        const marker = getSelectedMarker();
        if (marker) {
            marker.duration = parseFloat(markerDuration.value) || 1;
            Timeline.updateMarkerElement(marker);
        }
    });

    markerLanguage.addEventListener('change', () => {
        const marker = getSelectedMarker();
        if (marker) {
            marker.language = markerLanguage.value;
        }
    });

    // Удаление маркера
    btnDeleteMarker.addEventListener('click', () => {
        if (AppState.selectedMarkerId) {
            deleteMarker(AppState.selectedMarkerId);
        }
    });

    // Генерация TTS
    btnGenerateTts.addEventListener('click', () => {
        const marker = getSelectedMarker();
        if (marker && marker.text.trim()) {
            TTS.generate(marker);
        } else {
            notify('error', 'Введите текст реплики');
        }
    });

    // Ручная загрузка аудио
    markerAudioInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const marker = getSelectedMarker();
        if (!marker) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/upload', { method: 'POST', body: formData });
            const data = await resp.json();
            if (data.ok) {
                attachAudioToMarker(marker.id, data.url, data.filename);
                notify('success', 'Аудио прикреплено');
            }
        } catch (err) {
            notify('error', 'Ошибка загрузки аудио');
        }
    });

    // Прослушать прикреплённое аудио
    btnPlayAttached.addEventListener('click', () => {
        const marker = getSelectedMarker();
        if (marker && marker.audioUrl) {
            AudioEngine.playPreview(marker.audioUrl);
        }
    });

    // Удалить прикреплённое аудио
    btnRemoveAudio.addEventListener('click', () => {
        const marker = getSelectedMarker();
        if (marker) {
            marker.audioUrl = null;
            marker.audioFile = null;
            AudioEngine.removeAudio(marker.id);
            Timeline.updateMarkerElement(marker);
            updatePropertiesPanel();
        }
    });
}

function updatePropertiesPanel() {
    const noMarker = document.getElementById('no-marker-selected');
    const markerProps = document.getElementById('marker-properties');
    const marker = getSelectedMarker();

    if (!marker) {
        noMarker.hidden = false;
        markerProps.hidden = true;
        return;
    }

    noMarker.hidden = true;
    markerProps.hidden = false;

    // Заполняем поля
    document.getElementById('marker-text').value = marker.text;
    document.getElementById('marker-start').value = marker.startTime.toFixed(1);
    document.getElementById('marker-duration').value = marker.duration.toFixed(1);
    document.getElementById('marker-language').value = marker.language;

    // Аудио-информация
    const audioInfo = document.getElementById('marker-audio-info');
    const audioName = document.getElementById('attached-audio-name');
    if (marker.audioUrl) {
        audioInfo.hidden = false;
        audioName.textContent = marker.audioFile || 'аудио';
    } else {
        audioInfo.hidden = true;
    }

    // Сбрасываем варианты TTS при смене маркера
    document.getElementById('tts-variants').hidden = true;
    document.getElementById('tts-status').textContent = '';
}

function attachAudioToMarker(markerId, audioUrl, audioFile) {
    const marker = AppState.markers.find(m => m.id === markerId);
    if (!marker) return;

    marker.audioUrl = audioUrl;
    marker.audioFile = audioFile;
    AudioEngine.registerAudio(markerId, audioUrl, marker.startTime);
    Timeline.updateMarkerElement(marker);
    updatePropertiesPanel();
}

// ═══════════════════════════════════════════════════════════════════════
// TOOLBAR: Сохранение / Загрузка / Экспорт
// ═══════════════════════════════════════════════════════════════════════

function initToolbarActions() {
    document.getElementById('btn-save').addEventListener('click', saveProject);
    document.getElementById('btn-load').addEventListener('click', showLoadModal);
    document.getElementById('btn-export').addEventListener('click', exportProject);

    // Закрытие модальных окон
    document.querySelectorAll('.btn-close-modal').forEach(btn => {
        btn.addEventListener('click', () => {
            const modal = document.getElementById(btn.dataset.modal);
            if (modal) modal.hidden = true;
        });
    });
}

async function saveProject() {
    const name = document.getElementById('project-name').value.trim() || 'untitled';

    const projectData = {
        videoFile: AppState.videoFile,
        videoUrl: AppState.videoUrl,
        markers: AppState.markers.map(m => ({
            id: m.id,
            text: m.text,
            startTime: m.startTime,
            duration: m.duration,
            audioUrl: m.audioUrl,
            audioFile: m.audioFile,
            language: m.language
        }))
    };

    try {
        const resp = await fetch('/api/project/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, data: projectData })
        });
        const result = await resp.json();
        if (result.ok) {
            notify('success', `Проект "${name}" сохранён`);
        } else {
            notify('error', 'Ошибка сохранения');
        }
    } catch (err) {
        notify('error', `Ошибка: ${err.message}`);
    }
}

async function showLoadModal() {
    const modal = document.getElementById('modal-load');
    const list = document.getElementById('projects-list');
    modal.hidden = false;
    list.innerHTML = '<p class="hint">Загрузка...</p>';

    try {
        const resp = await fetch('/api/project/list');
        const data = await resp.json();

        if (!data.projects || data.projects.length === 0) {
            list.innerHTML = '<p class="hint">Нет сохранённых проектов</p>';
            return;
        }

        list.innerHTML = data.projects.map(p => `
            <div class="project-item" data-name="${p.name}">
                <span class="name">${p.name}</span>
                <span class="date">${p.savedAt ? new Date(p.savedAt).toLocaleString('ru') : ''}</span>
            </div>
        `).join('');

        // Обработчик клика на проект
        list.querySelectorAll('.project-item').forEach(item => {
            item.addEventListener('click', () => {
                loadProject(item.dataset.name);
                modal.hidden = true;
            });
        });
    } catch (err) {
        list.innerHTML = `<p class="hint" style="color:var(--accent)">Ошибка: ${err.message}</p>`;
    }
}

async function loadProject(name) {
    try {
        const resp = await fetch(`/api/project/load/${encodeURIComponent(name)}`);
        const data = await resp.json();

        if (!data.ok) throw new Error('Не удалось загрузить');

        const project = data.project.data;
        document.getElementById('project-name').value = data.project.name;

        // Очищаем текущее состояние
        AppState.markers = [];
        AppState.selectedMarkerId = null;
        AudioEngine.stopAll();
        Timeline.clearAll();

        // Восстанавливаем видео
        if (project.videoUrl) {
            const videoPlayer = document.getElementById('video-player');
            const placeholder = document.getElementById('video-placeholder');
            videoPlayer.src = project.videoUrl;
            videoPlayer.hidden = false;
            placeholder.hidden = true;
            AppState.videoFile = project.videoFile;
            AppState.videoUrl = project.videoUrl;

            videoPlayer.addEventListener('loadedmetadata', () => {
                AppState.videoDuration = videoPlayer.duration;
                document.getElementById('btn-play').disabled = false;
                document.getElementById('btn-add-marker').disabled = false;
                Timeline.init(videoPlayer.duration);

                // Восстанавливаем маркеры
                if (project.markers) {
                    project.markers.forEach(m => {
                        // Обновляем nextId чтобы не было конфликтов
                        const numId = parseInt(m.id.replace('marker_', ''));
                        if (numId >= AppState.nextId) AppState.nextId = numId + 1;

                        AppState.markers.push(m);
                        Timeline.renderMarker(m);
                        if (m.audioUrl) {
                            AudioEngine.registerAudio(m.id, m.audioUrl, m.startTime);
                        }
                    });
                }

                notify('success', `Проект "${name}" загружен`);
            }, { once: true });
        }
    } catch (err) {
        notify('error', `Ошибка загрузки: ${err.message}`);
    }
}

async function exportProject() {
    if (!AppState.videoFile) {
        notify('error', 'Сначала загрузите видео');
        return;
    }

    const markersWithAudio = AppState.markers.filter(m => m.audioFile);
    if (markersWithAudio.length === 0) {
        notify('error', 'Нет маркеров с аудио для экспорта');
        return;
    }

    const outputName = document.getElementById('project-name').value.trim() || 'export';

    notify('info', 'Экспорт запущен (ffmpeg)...');

    try {
        const resp = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video: AppState.videoFile,
                markers: markersWithAudio.map(m => ({
                    audio: m.audioFile,
                    startTime: m.startTime
                })),
                outputName: outputName
            })
        });
        const data = await resp.json();

        if (data.ok) {
            notify('success', `Экспорт готов: ${data.output}`);
        } else {
            notify('error', `Ошибка экспорта: ${data.error}`);
        }
    } catch (err) {
        notify('error', `Ошибка: ${err.message}`);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// УТИЛИТЫ
// ═══════════════════════════════════════════════════════════════════════

/**
 * Форматирование времени: 65.3 → "1:05.3"
 */
function formatTime(seconds) {
    if (isNaN(seconds)) return '0:00.0';
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(1);
    return `${mins}:${secs.padStart(4, '0')}`;
}

/**
 * Показать уведомление (type: info | success | error)
 */
function notify(type, message) {
    const container = document.getElementById('notifications');
    const el = document.createElement('div');
    el.className = `notification ${type}`;
    el.textContent = message;
    container.appendChild(el);

    // Автоудаление через 4 секунды
    setTimeout(() => {
        el.style.opacity = '0';
        setTimeout(() => el.remove(), 300);
    }, 4000);
}
