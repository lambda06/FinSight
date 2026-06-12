// Frontend Controller for FinSight RAG Dashboard

// State variables
let activeUserId = null;
let activeUserName = null;

// DOM Elements
const usersListEl = document.getElementById('users-list');
const profileStatsEl = document.getElementById('profile-stats');
const statDateRangeEl = document.getElementById('stat-date-range');
const statTransactionsEl = document.getElementById('stat-transactions');
const statAvgSpendEl = document.getElementById('stat-avg-spend');
const activeUserNameEl = document.getElementById('active-user-name');
const chatContainerEl = document.getElementById('chat-container');
const queryFormEl = document.getElementById('query-form');
const queryInputEl = document.getElementById('query-input');
const sendButtonEl = document.getElementById('send-button');
const welcomeMsgEl = document.getElementById('welcome-msg');

// Initial setup on load
document.addEventListener('DOMContentLoaded', () => {
    loadUsers();
    queryFormEl.addEventListener('submit', handleFormSubmit);
});

// Load all users from backend API
async function loadUsers() {
    try {
        const response = await fetch('/api/users');
        if (!response.ok) throw new Error('Failed to load users');
        const data = await response.json();
        
        usersListEl.innerHTML = '';
        data.users.forEach(user => {
            const card = document.createElement('div');
            card.className = 'user-card';
            card.dataset.id = user.id;
            card.dataset.name = user.name;
            card.dataset.range = user.date_range;
            card.dataset.tx = user.transactions;
            card.dataset.spend = Math.round(user.avg_spend);
            
            card.innerHTML = `
                <div class="user-card-header">
                    <span class="user-name">${user.name}</span>
                    <span class="user-id">${user.id}</span>
                </div>
                <div class="user-period">📅 ${user.date_range}</div>
            `;
            
            card.addEventListener('click', () => selectUser(card));
            usersListEl.appendChild(card);
        });
    } catch (error) {
        usersListEl.innerHTML = `<div class="badge badge-danger">Error: ${error.message}</div>`;
    }
}

// Handle User Selection
function selectUser(cardElement) {
    // Toggle active classes
    document.querySelectorAll('.user-card').forEach(el => el.classList.remove('active'));
    cardElement.classList.add('active');
    
    // Set active state
    activeUserId = cardElement.dataset.id;
    activeUserName = cardElement.dataset.name;
    
    // Update active user header details
    activeUserNameEl.textContent = `Active Session: ${activeUserName}`;
    const indicator = document.querySelector('.status-indicator');
    if (indicator) indicator.classList.add('active');
    
    // Populate stats overview card
    statDateRangeEl.textContent = cardElement.dataset.range;
    statTransactionsEl.textContent = cardElement.dataset.tx;
    statAvgSpendEl.textContent = `$${parseInt(cardElement.dataset.spend).toLocaleString()}/mo`;
    profileStatsEl.style.display = 'block';
    
    // Enable inputs
    queryInputEl.disabled = false;
    sendButtonEl.disabled = false;
    queryInputEl.placeholder = `Ask a question about ${activeUserName}'s spending...`;
    
    // Clear chat and hide welcome message
    welcomeMsgEl.style.display = 'none';
    
    // Remove old chat messages to start a clean user session
    const oldMessages = chatContainerEl.querySelectorAll('.chat-message');
    oldMessages.forEach(el => el.remove());
    
    // Append system greeting message
    appendSystemMessage(`Authorized access to <strong>${activeUserName}</strong> (${activeUserId}) active database. I'm ready to perform context-aware queries and safety checks.`);
}

// Handle Form Submission / Query Sent
async function handleFormSubmit(e) {
    e.preventDefault();
    const queryText = queryInputEl.value.trim();
    if (!queryText || !activeUserId) return;
    
    // Append User prompt to chat log
    appendMessage('user', queryText);
    queryInputEl.value = '';
    
    // Disable input while processing API call
    queryInputEl.disabled = true;
    sendButtonEl.disabled = true;
    
    // Append Loading Indicator
    const loader = appendLoadingIndicator();
    
    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: activeUserId, query: queryText })
        });
        
        // Remove loading spinner
        loader.remove();
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Failed to analyze query');
        }
        
        const result = await response.json();
        
        // Append AI response bubble
        appendAiResponse(result);
    } catch (error) {
        loader.remove();
        appendMessage('ai', `<div style="color:var(--status-error); font-weight:600;">System Error: ${error.message}</div>`);
    } finally {
        // Re-enable inputs
        queryInputEl.disabled = false;
        sendButtonEl.disabled = false;
        queryInputEl.focus();
    }
}

// Use a query suggestion tag directly
function useSuggestion(text) {
    if (!activeUserId) return;
    queryInputEl.value = text;
    queryFormEl.dispatchEvent(new Event('submit'));
}

// Append general raw message chat bubble
function appendMessage(sender, text) {
    const msg = document.createElement('div');
    msg.className = `chat-message ${sender}`;
    
    const senderName = sender === 'user' ? 'You' : 'FinSight Assistant';
    msg.innerHTML = `
        <span class="message-sender">${senderName}</span>
        <div class="message-bubble">${text}</div>
    `;
    
    chatContainerEl.appendChild(msg);
    chatContainerEl.scrollTop = chatContainerEl.scrollHeight;
    return msg;
}

// Append lightweight introductory system bubble
function appendSystemMessage(htmlText) {
    const msg = document.createElement('div');
    msg.className = 'chat-message ai';
    msg.innerHTML = `
        <span class="message-sender">System Log</span>
        <div class="message-bubble" style="background-color:rgba(16, 185, 129, 0.05); border-color:var(--status-success); border-style:dashed;">${htmlText}</div>
    `;
    chatContainerEl.appendChild(msg);
    chatContainerEl.scrollTop = chatContainerEl.scrollHeight;
}

// Append loader animation card
function appendLoadingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'chat-message ai loading-card';
    indicator.innerHTML = `
        <span class="message-sender">FinSight Assistant</span>
        <div class="message-bubble">
            <div class="loading-indicator">
                <div class="loading-spinner"></div>
                <span>Orchestrating RAG Pipeline (Stages 1-10)...</span>
            </div>
        </div>
    `;
    chatContainerEl.appendChild(indicator);
    chatContainerEl.scrollTop = chatContainerEl.scrollHeight;
    return indicator;
}

// Append and construct complete AI response package with charts and tags
function appendAiResponse(result) {
    const msg = document.createElement('div');
    msg.className = 'chat-message ai';
    
    // Convert markdown to clean HTML
    const formattedText = formatMarkdown(result.response);
    
    let htmlContent = `
        <span class="message-sender">FinSight Assistant</span>
        <div class="message-bubble">
            <div>${formattedText}</div>
    `;
    
    // Add visual charts if available
    if (result.visualizations && result.visualizations.length > 0) {
        htmlContent += `<div class="message-visualizations">`;
        result.visualizations.forEach(url => {
            // Append timestamp query parameter to bust browser cache on reload
            const cacheBustedUrl = `${url}?t=${new Date().getTime()}`;
            htmlContent += `
                <div class="chart-image-wrapper">
                    <img src="${cacheBustedUrl}" alt="Visualization output chart">
                </div>
            `;
        });
        htmlContent += `</div>`;
    }
    
    htmlContent += `</div>`; // Close bubble
    
    // Construct metadata tags footer (latency, model, cache status, safety flags)
    htmlContent += `<div class="message-metadata">`;
    
    // 1. Latency Badge
    htmlContent += `<span class="meta-badge">⚡ ${Math.round(result.latency_ms)} ms</span>`;
    
    // 2. Cache Badge
    htmlContent += `<span class="meta-badge">💾 ${result.cache_hit ? 'CACHE HIT' : 'CACHE MISS'}</span>`;
    
    // 3. Model Badge
    htmlContent += `<span class="meta-badge">🤖 ${result.model_used || 'None'}</span>`;
    
    // 4. Safety Guardrail Indicator Badge
    const status = result.status;
    const flags = result.guardrail_flags || [];
    
    if (status === 'guardrail_blocked') {
        htmlContent += `<span class="badge badge-danger">🚫 BLOCKED (${flags.join(', ')})</span>`;
    } else if (flags.includes('hallucination_flagged')) {
        htmlContent += `<span class="badge badge-warning">⚠️ FLAGGED (Hallucination)</span>`;
    } else {
        htmlContent += `<span class="badge badge-success">🛡️ Safe</span>`;
    }
    
    htmlContent += `</div>`; // Close metadata container
    
    msg.innerHTML = htmlContent;
    chatContainerEl.appendChild(msg);
    chatContainerEl.scrollTop = chatContainerEl.scrollHeight;
}

// Helper: Convert basic markdown elements to HTML
function formatMarkdown(text) {
    if (!text) return '';
    
    // Escape standard HTML tags first
    let escaped = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
        
    let lines = escaped.split('\n');
    let inTable = false;
    let tableHtml = '';
    let formattedLines = [];

    // 1. Format Markdown Tables
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i].trim();
        if (line.startsWith('|') && line.endsWith('|')) {
            if (line.includes('---') || line.includes(':---')) {
                // Skip layout separator lines
                continue;
            }
            if (!inTable) {
                inTable = true;
                tableHtml = '<table>';
            }
            
            let cells = line.split('|').slice(1, -1).map(c => c.trim());
            // First table line is assumed header
            let isHeader = !tableHtml.includes('<thead>');
            let cellTag = isHeader ? 'th' : 'td';
            let rowHtml = '<tr>' + cells.map(c => `<${cellTag}>${c}</${cellTag}>`).join('') + '</tr>';
            
            if (isHeader) {
                tableHtml += '<thead>' + rowHtml + '</thead><tbody>';
            } else {
                tableHtml += rowHtml;
            }
        } else {
            if (inTable) {
                inTable = false;
                tableHtml += '</tbody></table>';
                formattedLines.push(tableHtml);
            }
            formattedLines.push(line);
        }
    }
    if (inTable) {
        tableHtml += '</tbody></table>';
        formattedLines.push(tableHtml);
    }
    
    let html = formattedLines.join('\n');
    
    // 2. Bold text **bold** -> <strong>bold</strong>
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // 3. Newline characters -> <br> (but not inside table elements)
    html = html.split('\n').map(l => {
        const isTableElement = l.startsWith('<table') || 
                               l.startsWith('<thead') || 
                               l.startsWith('<tr') || 
                               l.startsWith('<td') || 
                               l.startsWith('<th') || 
                               l.startsWith('<tbody') || 
                               l.startsWith('</table') || 
                               l.startsWith('</tbody');
        if (isTableElement) {
            return l;
        }
        return l + '<br>';
    }).join('');
    
    return html;
}
