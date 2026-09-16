"use strict";

const model = document.querySelector("#model");
const input = document.querySelector("#message");
const send = document.querySelector("#send");
const newChat = document.querySelector("#new-chat");
const form = document.querySelector("#chat-form");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
let history = [];
let busy = false;
let ready = false;

function updateControls() {
  input.disabled = !ready || busy;
  model.disabled = !ready || busy;
  newChat.disabled = !ready || busy;
  send.disabled = !ready || busy || !input.value.trim();
  status.classList.toggle("busy", busy);
  form.setAttribute("aria-busy", String(busy));
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  updateControls();
}

function addMessage(role, content) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : model.selectedOptions[0].textContent;
  const body = document.createElement("div");
  body.className = "message-content";
  // Model output and user input are text, never executable HTML.
  body.textContent = content;
  article.append(label, body);
  messages.append(article);
  welcome.hidden = true;
  return article;
}

function scrollToComposer() {
  form.scrollIntoView({ block: "end" });
}

function resetChat() {
  history = [];
  messages.replaceChildren();
  welcome.hidden = false;
  error.hidden = true;
  input.value = "";
  status.textContent = "Ready when you are";
  resizeInput();
  input.focus();
}

async function readResponse(response) {
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("The server returned an unexpected response. Please try again.");
  }
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg).join(" ")
      : data.detail;
    throw new Error(typeof detail === "string" ? detail : "The request failed. Please try again.");
  }
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = input.value.trim();
  if (busy || !ready || !content) return;
  const pendingHistory = [...history, { role: "user", content }];
  const userMessage = addMessage("user", content);
  busy = true;
  error.hidden = true;
  input.value = "";
  status.textContent = "Waiting for a reply… The first message may take longer while the model loads.";
  resizeInput();
  scrollToComposer();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: model.value, messages: pendingHistory }),
    });
    const data = await readResponse(response);
    history = [...pendingHistory, { role: "assistant", content: data.reply }];
    addMessage("assistant", data.reply);
    status.textContent = "Ready when you are";
  } catch (failure) {
    userMessage.remove();
    welcome.hidden = history.length > 0;
    input.value = content;
    error.textContent = failure instanceof TypeError
      ? "Could not reach the server. Check your connection and try again."
      : failure.message;
    error.hidden = false;
    status.textContent = "Your message is ready to retry";
  } finally {
    busy = false;
    resizeInput();
    input.focus();
    scrollToComposer();
  }
});

input.addEventListener("input", resizeInput);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});
newChat.addEventListener("click", resetChat);
model.addEventListener("change", resetChat);
document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!ready || busy) return;
    input.value = button.dataset.prompt;
    resizeInput();
    input.focus();
  });
});

async function initialize() {
  try {
    const data = await readResponse(await fetch("/api/models"));
    model.replaceChildren();
    data.models.forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.id;
      option.textContent = `${entry.id} · ${entry.backend === "trtllm" ? "Local" : "Cloud"}`;
      model.append(option);
    });
    if (!data.models.length) throw new Error("No models are configured.");
    model.value = data.default;
    if (!model.value) model.selectedIndex = 0;
    ready = true;
    resetChat();
  } catch (failure) {
    status.textContent = "Connection unavailable";
    error.textContent = `${failure.message} Refresh the page to reconnect.`;
    error.hidden = false;
  }
}

initialize();
