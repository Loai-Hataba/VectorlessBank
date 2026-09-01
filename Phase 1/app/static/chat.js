/**
 * app/static/chat.js
 *
 * WHAT THIS FILE DOES
 * --------------------
 * Plain vanilla JS (no build step, no framework) that:
 *   1. Reads what the user types in the input box.
 *   2. Sends it to POST /api/chat/progress.
 *   3. Shows which step the assistant is on while it works, then
 *      renders the finished, checked answer.
 *
 * WHY PROGRESS
 * -------------
 * Every model call here runs on a local GPU, and one question takes
 * tens of seconds end to end. A spinner held that long reads as a
 * hung app. Naming the real step -- "Searching offers", "Checking my
 * answer" -- does not make the pipeline faster, it makes the wait
 * legible.
 *
 * WHY THE ANSWER IS NOT STREAMED
 * -------------------------------
 * It was, briefly. Two things killed it. The answer is written in the
 * fifth of six steps, so the first word did not appear until 110s
 * into a 125s turn -- the wait was over by then anyway. And the
 * output guardrail can only judge a FINISHED answer, so streaming
 * necessarily painted unchecked text on screen and then corrected it.
 * Nothing reaches the customer now until it has been approved.
 *
 * WHY SSE PARSED BY HAND
 * -----------------------
 * EventSource only issues GET requests and we need to POST the
 * question, so the event stream is read off fetch()'s body reader and
 * the (very simple) framing is parsed below.
 */

const chatWindow = document.getElementById("chat-window");
const composer = document.getElementById("composer");
const input = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");

const bankingTips = [
  "Tip: A budget is just a spending plan wearing a tie.",
  "Tip: Checking your balance is cheaper than guessing it.",
  "Tip: Small savings add up faster than loose change in a sofa.",
  "Tip: Before a purchase, give it one dramatic pause for thought.",
  "Tip: Your future self appreciates an emergency fund.",
];

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

  const turn = appendAssistantTurn();

  try {
    const response = await fetch("/api/chat/progress", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, session_id: sessionId }),
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      turn.fail(data.error || "Something went wrong.");
      return;
    }

    await readEventStream(response, (event, data) => {
      if (event === "stage") {
        turn.setStage(data.label, data.detail);
      } else if (event === "done") {
        turn.finish(data.answer, data.sources || []);
      } else if (event === "error") {
        turn.fail(data.message || "Something went wrong.");
      }
    });

    // The stream ended without a done or error frame -- the server died
    // mid-turn. Say so rather than leaving the progress line spinning
    // forever with no answer behind it.
    turn.failIfUnfinished("The connection ended before an answer arrived.");
  } catch (err) {
    turn.fail("Could not reach the server. Is the Flask app running?");
  } finally {
    setSending(false);
  }
});

/**
 * Read an SSE body and hand each complete frame to onEvent.
 * Frames are separated by a blank line; we only use `event:` and
 * `data:`, which is all the server emits.
 */
async function readEventStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);

      let name = "message";
      const dataLines = [];

      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) {
          name = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trim());
        }
      }

      if (dataLines.length === 0) continue;

      try {
        onEvent(name, JSON.parse(dataLines.join("\n")));
      } catch (e) {
        // A frame we cannot parse is not worth killing the turn over.
      }
    }
  }
}

function setSending(isSending) {
  sendButton.disabled = isSending;
  input.disabled = isSending;
}

function appendUserMessage(text) {
  const wrapper = document.createElement("div");
  wrapper.className = "message user";
  wrapper.innerHTML = `<div class="message-body"><div class="bubble">${escapeHtml(
    text
  )}</div></div>`;
  chatWindow.appendChild(wrapper);
  scrollToBottom();
}

/**
 * One assistant turn on screen, from "thinking" through to the final
 * checked answer. Returns the handful of operations the event loop
 * above needs, so the DOM bookkeeping stays in one place.
 */
function appendAssistantTurn() {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble typing";

  const status = document.createElement("div");
  status.className = "thinking-label";
  status.textContent = "Starting...";

  const tip = document.createElement("div");
  tip.className = "typing-tip";
  tip.textContent = bankingTips[0];

  const answerEl = document.createElement("div");
  answerEl.className = "answer-text";

  bubble.append(status, tip, answerEl);
  wrapper.appendChild(bubble);
  chatWindow.appendChild(wrapper);
  scrollToBottom();

  let tipIndex = 0;
  let tipTimer = window.setInterval(() => {
    tipIndex = (tipIndex + 1) % bankingTips.length;
    tip.textContent = bankingTips[tipIndex];
  }, 3200);

  let settled = false;

  function stopTips() {
    if (tipTimer !== null) {
      window.clearInterval(tipTimer);
      tipTimer = null;
    }
    tip.remove();
  }

  return {
    setStage(label, detail) {
      status.textContent = detail ? `${label}: ${detail}` : label;
    },

    finish(answer, sources) {
      settled = true;
      stopTips();
      status.remove();
      bubble.classList.remove("typing");

      // This is the guardrail-approved text -- possibly with a
      // disclaimer appended, occasionally a refusal. It is what the
      // customer was actually told, and the only text they ever see.
      answerEl.textContent = answer;

      if (sources.length > 0) {
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

      scrollToBottom();
    },

    fail(message) {
      settled = true;
      stopTips();
      status.remove();
      bubble.classList.remove("typing");
      bubble.classList.add("error-bubble");
      answerEl.textContent = message;
      scrollToBottom();
    },

    failIfUnfinished(message) {
      if (!settled) this.fail(message);
    },
  };
}

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
