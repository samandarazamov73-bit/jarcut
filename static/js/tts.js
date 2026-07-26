/**
 * CapCut-Lite — Модуль TTS (tts.js)
 * 
 * Отвечает за:
 * - Отправку текста на сервер для генерации озвучки (Gemini TTS)
 * - Отображение 2-3 вариантов голоса с возможностью прослушать
 * - Статус каждого варианта (loading / ready / error)
 * - Выбор варианта → прикрепление к маркеру
 * 
 * Зависит от: AppState, attachAudioToMarker, notify (из app.js)
 */

const TTS = (() => {

    // ─── Генерация вариантов ────────────────────────────────────────────

    /**
     * Запускает генерацию нескольких вариантов озвучки для маркера.
     * @param {Object} marker - объект маркера из AppState.markers
     */
    async function generate(marker) {
        const statusEl = document.getElementById('tts-status');
        const variantsSection = document.getElementById('tts-variants');
        const container = document.getElementById('variants-container');

        // Показываем статус загрузки
        statusEl.textContent = '⏳ Генерация вариантов озвучки...';
        statusEl.className = 'status-text loading';

        // Очищаем предыдущие варианты
        container.innerHTML = '';
        variantsSection.hidden = false;

        // Показываем заглушки (loading state) для каждого варианта
        const numVariants = 3;
        for (let i = 0; i < numVariants; i++) {
            const placeholder = createVariantPlaceholder(i);
            container.appendChild(placeholder);
        }

        try {
            // Запрос к серверу
            const resp = await fetch('/api/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: marker.text,
                    variants: numVariants,
                    language: marker.language
                })
            });

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${resp.status}`);
            }

            const data = await resp.json();

            if (!data.ok || !data.variants) {
                throw new Error('Некорректный ответ сервера');
            }

            // Обновляем статус
            const readyCount = data.variants.filter(v => v.status === 'ready').length;
            const errorCount = data.variants.filter(v => v.status === 'error').length;

            if (readyCount > 0) {
                statusEl.textContent = `✓ Готово: ${readyCount} вариант(ов)${errorCount > 0 ? `, ${errorCount} с ошибкой` : ''}`;
                statusEl.className = 'status-text success';
            } else {
                statusEl.textContent = `✗ Все варианты с ошибками`;
                statusEl.className = 'status-text error';
            }

            // Обновляем UI вариантов
            container.innerHTML = '';
            data.variants.forEach((variant, index) => {
                const el = createVariantElement(variant, index, marker.id);
                container.appendChild(el);
            });

        } catch (err) {
            statusEl.textContent = `✗ Ошибка: ${err.message}`;
            statusEl.className = 'status-text error';
            container.innerHTML = '';
            notify('error', `TTS ошибка: ${err.message}`);
        }
    }

    // ─── UI: заглушка варианта (loading) ────────────────────────────────

    function createVariantPlaceholder(index) {
        const el = document.createElement('div');
        el.className = 'variant-item loading';
        el.innerHTML = `
            <span class="variant-voice">Голос ${index + 1}</span>
            <span class="variant-status">⏳ Генерация...</span>
        `;
        return el;
    }

    // ─── UI: элемент варианта (готовый) ─────────────────────────────────

    function createVariantElement(variant, index, markerId) {
        const el = document.createElement('div');

        if (variant.status === 'error') {
            el.className = 'variant-item error';
            el.innerHTML = `
                <span class="variant-voice">${variant.voice_name || `Голос ${index + 1}`}</span>
                <span class="variant-status" title="${variant.error || ''}">✗ Ошибка</span>
            `;
            return el;
        }

        // Вариант готов
        el.className = 'variant-item';
        el.innerHTML = `
            <span class="variant-voice">${variant.voice_name || `Голос ${index + 1}`}</span>
            <audio controls preload="auto" src="${variant.audio_url}"></audio>
            <button class="btn-select btn-small" title="Выбрать этот вариант">✓ Выбрать</button>
        `;

        // Кнопка "Выбрать" → прикрепить к маркеру
        const btnSelect = el.querySelector('.btn-select');
        btnSelect.addEventListener('click', () => {
            selectVariant(el, variant, markerId);
        });

        return el;
    }

    // ─── Выбор варианта ─────────────────────────────────────────────────

    function selectVariant(element, variant, markerId) {
        // Снимаем выделение с других
        const container = document.getElementById('variants-container');
        container.querySelectorAll('.variant-item').forEach(el => {
            el.classList.remove('selected');
        });

        // Выделяем выбранный
        element.classList.add('selected');

        // Прикрепляем аудио к маркеру
        // audio_url содержит путь типа "/uploads/tts_abc123.wav"
        // audioFile — имя файла для экспорта
        const audioFile = variant.audio_url.split('/').pop();
        attachAudioToMarker(markerId, variant.audio_url, audioFile);

        notify('success', `Голос "${variant.voice_name}" прикреплён к маркеру`);
    }

    // ─── Публичный API ──────────────────────────────────────────────────

    return {
        generate
    };

})();
