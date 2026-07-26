/**
 * CapCut-Lite — Модуль таймлайна (timeline.js)
 * 
 * Отвечает за:
 * - Рисование шкалы времени (ruler) с делениями по секундам
 * - Отображение маркеров на дорожке
 * - Drag & Drop маркеров (перемещение по времени)
 * - Resize маркеров (изменение длительности)
 * - Playhead (полоса текущего времени)
 * - Zoom (масштабирование)
 * - Клик по ruler → перемотка видео
 * 
 * Зависит от: AppState, selectMarker, seekTo (из app.js)
 */

const Timeline = (() => {
    // ─── Настройки ──────────────────────────────────────────────────────

    let duration = 0;           // длительность видео (сек)
    let pixelsPerSecond = 80;   // масштаб: пикселей на секунду
    let minPPS = 20;            // минимальный масштаб
    let maxPPS = 200;           // максимальный масштаб

    // DOM-элементы (кэшируем после init)
    let wrapper = null;
    let ruler = null;
    let playhead = null;
    let trackContent = null;

    // Состояние drag
    let dragState = null;  // { markerId, type: 'move'|'resize', startX, startLeft, startWidth }

    // ─── Инициализация ──────────────────────────────────────────────────

    function init(videoDuration) {
        duration = videoDuration;

        wrapper = document.getElementById('timeline-wrapper');
        ruler = document.getElementById('time-ruler');
        playhead = document.getElementById('playhead');
        trackContent = document.getElementById('track-content-markers');

        renderRuler();
        setupRulerClick();
        setupZoomButtons();
        setupDrag();

        // Устанавливаем ширину контейнера дорожки
        updateTrackWidth();
    }

    // ─── Шкала времени (ruler) ──────────────────────────────────────────

    function renderRuler() {
        if (!ruler) return;
        ruler.innerHTML = '';

        const totalWidth = duration * pixelsPerSecond;
        ruler.style.width = `${totalWidth}px`;

        // Определяем шаг делений в зависимости от масштаба
        let majorStep, minorStep;
        if (pixelsPerSecond >= 100) {
            majorStep = 1;   // каждую секунду крупная метка
            minorStep = 0.5; // полусекунда
        } else if (pixelsPerSecond >= 50) {
            majorStep = 2;
            minorStep = 1;
        } else if (pixelsPerSecond >= 30) {
            majorStep = 5;
            minorStep = 1;
        } else {
            majorStep = 10;
            minorStep = 5;
        }

        for (let t = 0; t <= duration; t += minorStep) {
            const isMajor = (t % majorStep === 0);
            const mark = document.createElement('div');
            mark.className = `ruler-mark${isMajor ? ' major' : ''}`;
            mark.style.left = `${t * pixelsPerSecond}px`;

            if (isMajor) {
                mark.textContent = formatTimeShort(t);
            }

            ruler.appendChild(mark);
        }
    }

    function formatTimeShort(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        if (mins === 0) return `${secs}s`;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    // ─── Клик по шкале → перемотка ──────────────────────────────────────

    function setupRulerClick() {
        ruler.addEventListener('click', (e) => {
            const rect = ruler.getBoundingClientRect();
            const x = e.clientX - rect.left + wrapper.scrollLeft;
            const time = x / pixelsPerSecond;
            seekTo(Math.max(0, Math.min(time, duration)));
        });
    }

    // ─── Zoom ───────────────────────────────────────────────────────────

    function setupZoomButtons() {
        document.getElementById('btn-zoom-in').addEventListener('click', () => {
            pixelsPerSecond = Math.min(maxPPS, pixelsPerSecond + 20);
            refresh();
        });

        document.getElementById('btn-zoom-out').addEventListener('click', () => {
            pixelsPerSecond = Math.max(minPPS, pixelsPerSecond - 20);
            refresh();
        });
    }

    function refresh() {
        document.getElementById('zoom-level').textContent = 
            `Масштаб: ${Math.round(pixelsPerSecond / 80 * 100)}%`;
        renderRuler();
        updateTrackWidth();
        rerenderAllMarkers();
        updatePlayhead(document.getElementById('video-player').currentTime);
    }

    function updateTrackWidth() {
        const totalWidth = duration * pixelsPerSecond;
        trackContent.style.width = `${totalWidth}px`;
    }

    // ─── Playhead ───────────────────────────────────────────────────────

    function updatePlayhead(currentTime) {
        if (!playhead) return;
        const left = currentTime * pixelsPerSecond;
        playhead.style.left = `${left + 80}px`; // +80 для label трека
    }

    // ─── Маркеры: рендер ────────────────────────────────────────────────

    function renderMarker(marker) {
        const el = document.createElement('div');
        el.className = 'marker';
        el.id = `marker-el-${marker.id}`;
        el.dataset.markerId = marker.id;

        // Позиционирование
        el.style.left = `${marker.startTime * pixelsPerSecond}px`;
        el.style.width = `${marker.duration * pixelsPerSecond}px`;

        // Текст (обрезаем если длинный)
        const label = document.createElement('span');
        label.className = 'marker-label';
        label.textContent = marker.text.substring(0, 30);
        el.appendChild(label);

        // Ручка ресайза
        const handle = document.createElement('div');
        handle.className = 'marker-resize-handle';
        el.appendChild(handle);

        // Класс если есть аудио
        if (marker.audioUrl) {
            el.classList.add('has-audio');
        }

        // Клик → выбор
        el.addEventListener('mousedown', (e) => {
            if (e.target === handle) return; // ресайз обрабатывается отдельно
            selectMarker(marker.id);
        });

        trackContent.appendChild(el);
    }

    function updateMarkerElement(marker) {
        const el = document.getElementById(`marker-el-${marker.id}`);
        if (!el) return;

        el.style.left = `${marker.startTime * pixelsPerSecond}px`;
        el.style.width = `${marker.duration * pixelsPerSecond}px`;

        // Обновляем текст
        const label = el.querySelector('.marker-label');
        if (label) label.textContent = marker.text.substring(0, 30);

        // Обновляем класс аудио
        el.classList.toggle('has-audio', !!marker.audioUrl);
    }

    function removeMarker(id) {
        const el = document.getElementById(`marker-el-${id}`);
        if (el) el.remove();
    }

    function highlightMarker(id) {
        // Снимаем выделение со всех
        trackContent.querySelectorAll('.marker').forEach(el => {
            el.classList.remove('selected');
        });

        // Выделяем нужный
        const el = document.getElementById(`marker-el-${id}`);
        if (el) el.classList.add('selected');
    }

    function rerenderAllMarkers() {
        trackContent.innerHTML = '';
        AppState.markers.forEach(m => renderMarker(m));
        if (AppState.selectedMarkerId) {
            highlightMarker(AppState.selectedMarkerId);
        }
    }

    function clearAll() {
        if (trackContent) trackContent.innerHTML = '';
    }

    // ─── Drag & Drop + Resize ───────────────────────────────────────────

    function setupDrag() {
        // Используем делегирование событий на trackContent
        document.addEventListener('mousedown', (e) => {
            const markerEl = e.target.closest('.marker');
            if (!markerEl) return;
            if (!trackContent.contains(markerEl)) return;

            const markerId = markerEl.dataset.markerId;
            const isResize = e.target.classList.contains('marker-resize-handle');

            e.preventDefault();
            selectMarker(markerId);

            dragState = {
                markerId: markerId,
                type: isResize ? 'resize' : 'move',
                startX: e.clientX,
                startLeft: parseFloat(markerEl.style.left),
                startWidth: parseFloat(markerEl.style.width),
                element: markerEl
            };

            markerEl.classList.add('dragging');
        });

        document.addEventListener('mousemove', (e) => {
            if (!dragState) return;

            const dx = e.clientX - dragState.startX;
            const marker = AppState.markers.find(m => m.id === dragState.markerId);
            if (!marker) return;

            if (dragState.type === 'move') {
                // Перемещение
                let newLeft = Math.max(0, dragState.startLeft + dx);
                const maxLeft = duration * pixelsPerSecond - parseFloat(dragState.element.style.width);
                newLeft = Math.min(newLeft, maxLeft);

                dragState.element.style.left = `${newLeft}px`;
                marker.startTime = newLeft / pixelsPerSecond;
            } else {
                // Resize
                let newWidth = Math.max(pixelsPerSecond * 0.2, dragState.startWidth + dx); // мин 0.2 сек
                const maxWidth = (duration - marker.startTime) * pixelsPerSecond;
                newWidth = Math.min(newWidth, maxWidth);

                dragState.element.style.width = `${newWidth}px`;
                marker.duration = newWidth / pixelsPerSecond;
            }
        });

        document.addEventListener('mouseup', () => {
            if (!dragState) return;

            dragState.element.classList.remove('dragging');

            // Обновляем UI панели свойств
            const marker = AppState.markers.find(m => m.id === dragState.markerId);
            if (marker) {
                // Обновляем позицию аудио если есть
                if (marker.audioUrl) {
                    AudioEngine.updateTiming(marker.id, marker.startTime);
                }
                updatePropertiesPanel();
            }

            dragState = null;
        });
    }

    // ─── Публичный API ──────────────────────────────────────────────────

    return {
        init,
        renderMarker,
        updateMarkerElement,
        removeMarker,
        highlightMarker,
        updatePlayhead,
        clearAll,
        // Для доступа извне (например для скролла к позиции)
        getPixelsPerSecond: () => pixelsPerSecond
    };
})();
