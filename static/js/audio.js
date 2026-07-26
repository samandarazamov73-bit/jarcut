/**
 * CapCut-Lite — Модуль аудио-движка (audio.js)
 * 
 * Отвечает за:
 * - Загрузку и кэширование аудио-буферов (Web Audio API)
 * - Синхронизированное воспроизведение аудио-маркеров с video.currentTime
 * - Запуск/остановку/пауза аудио при play/pause/seek видео
 * - Простое превью (прослушивание одного файла)
 * 
 * Как работает синхронизация:
 * При каждом вызове syncWithVideo(currentTime) проверяем все маркеры:
 * - если currentTime попал в [startTime, startTime+duration] и маркер НЕ играет → запускаем
 * - если currentTime вне диапазона и маркер играет → останавливаем
 * 
 * Зависит от: AppState (из app.js)
 */

const AudioEngine = (() => {

    // ─── Состояние ──────────────────────────────────────────────────────

    let audioContext = null;      // AudioContext (создаётся лениво)
    const audioBuffers = {};      // { markerId: AudioBuffer }
    const activeSources = {};     // { markerId: { source, startedAt, offset } }
    const audioUrls = {};         // { markerId: url }
    const audioTimings = {};      // { markerId: startTime }

    let isEngineActive = false;   // true когда видео играет
    let lastSyncTime = 0;        // последнее известное время видео

    // ─── Инициализация AudioContext ─────────────────────────────────────

    function getContext() {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        // Возобновляем если suspended (требование браузера)
        if (audioContext.state === 'suspended') {
            audioContext.resume();
        }
        return audioContext;
    }

    // ─── Регистрация аудио для маркера ──────────────────────────────────

    /**
     * Регистрирует аудио-файл для маркера. Загружает и декодирует буфер.
     * @param {string} markerId 
     * @param {string} url - URL аудиофайла (/uploads/...)
     * @param {number} startTime - время начала на таймлайне (сек)
     */
    async function registerAudio(markerId, url, startTime) {
        audioUrls[markerId] = url;
        audioTimings[markerId] = startTime;

        // Загружаем и декодируем аудио
        try {
            const ctx = getContext();
            const response = await fetch(url);
            const arrayBuffer = await response.arrayBuffer();
            const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
            audioBuffers[markerId] = audioBuffer;
        } catch (err) {
            console.warn(`AudioEngine: не удалось загрузить аудио для ${markerId}:`, err);
        }
    }

    // ─── Обновление тайминга (после перетаскивания маркера) ──────────────

    function updateTiming(markerId, newStartTime) {
        audioTimings[markerId] = newStartTime;
        // Если маркер сейчас играет — перезапустим при следующем sync
        stopMarker(markerId);
    }

    // ─── Удаление аудио маркера ─────────────────────────────────────────

    function removeAudio(markerId) {
        stopMarker(markerId);
        delete audioBuffers[markerId];
        delete audioUrls[markerId];
        delete audioTimings[markerId];
    }

    // ─── Синхронизация с видео ──────────────────────────────────────────

    /**
     * Вызывается из video.ontimeupdate (≈4 раза/сек).
     * Проверяет какие маркеры должны играть в текущий момент.
     */
    function syncWithVideo(currentTime) {
        if (!isEngineActive) return;
        lastSyncTime = currentTime;

        // Перебираем все маркеры с аудио
        for (const marker of AppState.markers) {
            if (!marker.audioUrl) continue;
            if (!audioBuffers[marker.id]) continue;

            const markerStart = audioTimings[marker.id] ?? marker.startTime;
            const buffer = audioBuffers[marker.id];
            const markerEnd = markerStart + buffer.duration;

            const isInRange = currentTime >= markerStart && currentTime < markerEnd;
            const isPlaying = !!activeSources[marker.id];

            if (isInRange && !isPlaying) {
                // Нужно запустить маркер с правильным offset
                const offset = currentTime - markerStart;
                playMarker(marker.id, offset);
            } else if (!isInRange && isPlaying) {
                // Маркер вышел из диапазона — остановить
                stopMarker(marker.id);
            }
        }
    }

    // ─── Воспроизведение маркера ────────────────────────────────────────

    function playMarker(markerId, offset = 0) {
        const buffer = audioBuffers[markerId];
        if (!buffer) return;

        // Останавливаем если уже играет
        stopMarker(markerId);

        const ctx = getContext();
        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(ctx.destination);

        // Запускаем с offset
        source.start(0, offset);

        activeSources[markerId] = {
            source: source,
            startedAt: ctx.currentTime - offset,
            offset: offset
        };

        // Автоочистка при завершении
        source.onended = () => {
            delete activeSources[markerId];
        };
    }

    function stopMarker(markerId) {
        const active = activeSources[markerId];
        if (active) {
            try {
                active.source.stop();
            } catch (e) {
                // ignore — может быть уже остановлен
            }
            delete activeSources[markerId];
        }
    }

    // ─── Управление движком (play/pause/stop) ───────────────────────────

    /**
     * Вызывается при нажатии Play на видео.
     */
    function resumeAll(currentTime) {
        getContext(); // убеждаемся что AudioContext активен
        isEngineActive = true;
        lastSyncTime = currentTime;
        // Сразу синхронизируем — запустит нужные маркеры
        syncWithVideo(currentTime);
    }

    /**
     * Вызывается при нажатии Pause на видео.
     */
    function pauseAll() {
        isEngineActive = false;
        // Останавливаем все играющие маркеры
        for (const markerId of Object.keys(activeSources)) {
            stopMarker(markerId);
        }
    }

    /**
     * Полная остановка (конец видео, смена проекта).
     */
    function stopAll() {
        isEngineActive = false;
        for (const markerId of Object.keys(activeSources)) {
            stopMarker(markerId);
        }
    }

    // ─── Превью (прослушивание одного файла без привязки к видео) ────────

    let previewSource = null;

    /**
     * Простое воспроизведение аудио для превью (кнопка ▶ в панели свойств).
     */
    async function playPreview(url) {
        // Останавливаем предыдущее превью
        if (previewSource) {
            try { previewSource.stop(); } catch(e) {}
            previewSource = null;
        }

        try {
            const ctx = getContext();
            const response = await fetch(url);
            const arrayBuffer = await response.arrayBuffer();
            const buffer = await ctx.decodeAudioData(arrayBuffer);

            previewSource = ctx.createBufferSource();
            previewSource.buffer = buffer;
            previewSource.connect(ctx.destination);
            previewSource.start(0);

            previewSource.onended = () => {
                previewSource = null;
            };
        } catch (err) {
            console.warn('AudioEngine.playPreview error:', err);
        }
    }

    // ─── Публичный API ──────────────────────────────────────────────────

    return {
        registerAudio,
        updateTiming,
        removeAudio,
        syncWithVideo,
        resumeAll,
        pauseAll,
        stopAll,
        playPreview
    };

})();
