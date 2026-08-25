/**
 * app/static/chat.js
 *
 * WHAT THIS FILE DOES
 * --------------------
 * Plain vanilla JS (no build step, no framework) that:
 *   1. Reads what the user types in the input box.
 *   2. Sends it to POST /api/chat as JSON.
 *   3. Renders the assistant's reply (and which sources it used) into the chat window.
 *
 * WHY VANILLA JS
 * ---------------
 * The frontend is explicitly not the focus of this project -- the
 * retrieval/RAG system is. A single plain JS file that any browser
 * can run with zero tooling keeps this piece simple and out of the way.
 */

const chatWindow = document.getElementById("chat-window");
const composer = document.getElementById("composer");
const input = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");

function getSessionId() {
  const sessionId = crypto.randomUUID();
  localStorage.setItem("session_id", sessionId);
  return sessionId;
}

const sessionId = getSessionId();

composer.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = input.value.trim();
  if (!question) return;

  appendUserMessage(question);
  input.value = "";
  setSending(true);

  const typingEl = appendTypingIndicator();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, session_id: sessionId }),
    });

    const data = await response.json();
    typingEl.remove();

    if (!response.ok) {
      appendAssistantMessage(data.error || "Something went wrong.", [], true);
    } else {
      appendAssistantMessage(data.answer, data.sources || [], false);
    }
  } catch (err) {
    typingEl.remove();
    appendAssistantMessage(
      "Could not reach the server. Is the Flask app running?",
      [],
      true
    );
  } finally {
    setSending(false);
  }
});

function setSending(isSending) {
  sendButton.disabled = isSending;
  input.disabled = isSending;
}

function appendUserMessage(text) {
  const wrapper = document.createElement("div");
  wrapper.className = "message user";
  wrapper.innerHTML = `<div class="message-body"><div class="bubble">${escapeHtml(text)}</div></div>`;
  chatWindow.appendChild(wrapper);
  scrollToBottom();
}

function appendAssistantMessage(text, sources, isError) {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble" + (isError ? " error-bubble" : "");
  bubble.textContent = text;
  wrapper.appendChild(bubble);
  if (sources && sources.length > 0) {
    const sourcesEl = document.createElement("div");
    sourcesEl.className = "sources";
    const label = document.createElement("span");
    label.className = "sources-label";
    label.textContent = "Sourced from";
    sourcesEl.appendChild(label);
    sources.forEach((s) => {
      const stamp = document.createElement("span");
      stamp.className = "source-stamp";
      stamp.textContent = s.title;
      sourcesEl.appendChild(stamp);
    });
    bubble.appendChild(sourcesEl);
  }
  chatWindow.appendChild(wrapper);
  scrollToBottom();
}

function appendTypingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";
  wrapper.innerHTML = `<div class="bubble typing">thinking...</div>`;
  chatWindow.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
