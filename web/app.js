"use strict";

const model = document.querySelector("#model");
const chatbotSelect = document.querySelector("#chatbot");
const input = document.querySelector("#message");
const send = document.querySelector("#send");
const newChat = document.querySelector("#new-chat");
const form = document.querySelector("#chat-form");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const heading = document.querySelector("#welcome-heading");
const introduction = document.querySelector("#introduction");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
let activeChatbot = null;
let sessionId = null;
let idleDeadline = 0;
let idleTimeout = 60;
let busy = false;
let ready = false;

function updateControls() {
  input.disabled = !ready || busy || !activeChatbot;
  model.disabled = !ready || busy;
  chatbotSelect.disabled = !ready || busy;
  newChat.disabled = !ready || busy;
  send.disabled = input.disabled || !input.value.trim();
  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.disabled = input.disabled;
  });
  status.classList.toggle("busy", busy);
  form.setAttribute("aria-busy", String(busy));
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  updateControls();
}

function scrollToLatest() {
  window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "instant" });
}

function addMessage(role, content) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : activeChatbot.name;
  const body = document.createElement("div");
  body.className = "message-content";
  body.textContent = content;
  article.append(label, body);
  messages.append(article);
  welcome.hidden = true;
  return article;
}

function renderConversation(data) {
  activeChatbot = data.chatbot;
  chatbotSelect.value = String(activeChatbot.id);
  messages.replaceChildren();
  welcome.hidden = data.messages.length > 0;
  heading.textContent = `Meet ${activeChatbot.name}`;
  introduction.textContent = "Ask a question, explore an idea, or pick up where you left off.";
  data.messages.forEach((message) => addMessage(message.role, message.content));
  document.title = `${activeChatbot.name} · Chatbot`;
}

function activateSession(data) {
  sessionId = data.session_id;
  idleTimeout = data.idle_timeout_seconds;
  idleDeadline = Date.now() + idleTimeout * 1000;
  status.textContent = `${activeChatbot.name} · Ready when you are`;
}

function checkIdle() {
  if (!busy && sessionId && Date.now() >= idleDeadline) {
    sessionId = null;
    status.textContent = "Session ended after a minute of inactivity. Send a message to resume.";
  }
}
setInterval(checkIdle, 1000);
document.addEventListener("visibilitychange", checkIdle);

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
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

function showError(failure) {
  error.textContent = failure instanceof TypeError
    ? "Could not reach the server. Check your connection and try again."
    : failure.message;
  error.hidden = false;
}

function addChatbotOption(chatbot) {
  if ([...chatbotSelect.options].some((option) => option.value === String(chatbot.id))) return;
  const option = document.createElement("option");
  option.value = chatbot.id;
  option.textContent = chatbot.name;
  chatbotSelect.append(option);
}

async function openChatbot(id) {
  const data = await api(`/api/chatbots/${id}/open`, {});
  renderConversation(data);
  if ([...model.options].some((option) => option.value === activeChatbot.model)) {
    model.value = activeChatbot.model;
  }
  input.value = "";
  activateSession(data);
}

async function chooseChatbot(create) {
  if (busy || !ready) return;
  busy = true;
  error.hidden = true;
  status.textContent = create ? "Creating your chatbot…" : "Opening your chatbot…";
  updateControls();
  try {
    let id = chatbotSelect.value;
    if (create) {
      const chatbot = await api("/api/chatbots", { model: model.value });
      addChatbotOption(chatbot);
      id = chatbot.id;
    }
    await openChatbot(id);
  } catch (failure) {
    chatbotSelect.value = activeChatbot ? String(activeChatbot.id) : "";
    showError(failure);
    status.textContent = "Could not open the chatbot. Please try again.";
  } finally {
    busy = false;
    resizeInput();
    if (activeChatbot) {
      input.focus();
      scrollToLatest();
    }
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = input.value.trim();
  if (busy || !ready || !activeChatbot || !content) return;
  checkIdle();
  const userMessage = addMessage("user", content);
  busy = true;
  error.hidden = true;
  input.value = "";
  status.textContent = sessionId ? "Waiting for a reply…" : "Resuming your session…";
  resizeInput();
  scrollToLatest();
  try {
    const data = await api(`/api/chatbots/${activeChatbot.id}/messages`, {
      content, model: model.value, session_id: sessionId,
    });
    renderConversation(data);
    activateSession(data);
  } catch (failure) {
    userMessage.remove();
    welcome.hidden = messages.childElementCount > 0;
    input.value = content;
    showError(failure);
    status.textContent = "Your message is ready to retry";
  } finally {
    busy = false;
    resizeInput();
    input.focus();
    scrollToLatest();
  }
});

input.addEventListener("input", resizeInput);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});
newChat.addEventListener("click", () => chooseChatbot(true));
chatbotSelect.addEventListener("change", () => chooseChatbot(false));
model.addEventListener("change", () => {
  if (activeChatbot) status.textContent = "Model changed. Your conversation will continue with the next message.";
});
document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!ready || busy || !activeChatbot) return;
    input.value = button.dataset.prompt;
    resizeInput();
    input.focus();
  });
});

async function initialize() {
  try {
    const [settings, saved] = await Promise.all([api("/api/models"), api("/api/chatbots")]);
    model.replaceChildren();
    settings.models.forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.id;
      option.textContent = `${entry.id} · ${entry.backend === "trtllm" ? "Local" : "Cloud"}`;
      model.append(option);
    });
    if (!settings.models.length) throw new Error("No models are configured.");
    model.value = settings.default;
    if (!model.value) model.selectedIndex = 0;
    saved.chatbots.forEach(addChatbotOption);
    idleTimeout = saved.idle_timeout_seconds;
    if (saved.created) {
      await openChatbot(saved.chatbots[0].id);
    } else {
      heading.textContent = "Welcome back";
      introduction.textContent = "Choose a saved chatbot above, or create a new one to start fresh.";
      status.textContent = "Choose a chatbot or create a new one";
    }
    ready = true;
    resizeInput();
    if (activeChatbot) input.focus();
  } catch (failure) {
    status.textContent = "Connection unavailable";
    showError(failure);
  }
}

initialize();
