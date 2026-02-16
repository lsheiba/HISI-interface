/**
 * Shared utility functions for ASR interface.
 * Contains common code used by both recording and upload modes.
 */

window.formatTime = function(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toFixed(3).padStart(6, '0')}`;
};

window.formatDuration = function(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs.toFixed(1)}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs.toFixed(1)}s`;
    } else {
        return `${secs.toFixed(1)}s`;
    }
};

window.updateSpeakerLegend = function(segments, legendItemsId, legendContainerId) {
    const uniqueSpeakers = new Set();
    segments.forEach(seg => {
        if (seg.speaker) uniqueSpeakers.add(seg.speaker);
    });
    
    const legendItems = document.getElementById(legendItemsId);
    const legendContainer = document.getElementById(legendContainerId);
    if (!legendItems || !legendContainer) return;
    
    if (uniqueSpeakers.size > 0) {
        legendContainer.style.display = 'flex';
        legendItems.innerHTML = '';
        uniqueSpeakers.forEach(speaker => {
            const speakerClass = speaker.toLowerCase().replace(/\s+/g, '-');
            const item = document.createElement('span');
            item.className = `speaker-legend-item speaker-${speakerClass}`;
            item.textContent = speaker;
            legendItems.appendChild(item);
        });
    } else {
        legendContainer.style.display = 'none';
    }
};

window.getSpeakerClass = function(speaker) {
    if (!speaker) return '';
    return speaker.toLowerCase().replace(/\s+/g, '-');
};

window.createSpeakerCell = function(segment, diarizationEnabled) {
    if (!diarizationEnabled) return '';
    
    if (segment.speaker) {
        const speakerClass = getSpeakerClass(segment.speaker);
        return `<td class="speaker-cell speaker-${speakerClass}">${segment.speaker}</td>`;
    }
    return '<td class="speaker-cell">-</td>';
};
