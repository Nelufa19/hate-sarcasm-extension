const statusEl = document.getElementById("status");
const toggleEl = document.getElementById("enabledToggle");

fetch("http://localhost:8000/health")
  .then((res) => res.json())
  .then((data) => {
    statusEl.textContent = `online (${data.models_loaded.join(", ")})`;
    statusEl.className = "ok";
  })
  .catch(() => {
    statusEl.textContent = "unreachable";
    statusEl.className = "down";
  });

// Enabled defaults to true if never set before.
chrome.storage.local.get({ enabled: true }, ({ enabled }) => {
  toggleEl.checked = enabled;
});

toggleEl.addEventListener("change", () => {
  chrome.storage.local.set({ enabled: toggleEl.checked });
});
