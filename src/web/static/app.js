const form = document.querySelector("#prediction-form");
const result = document.querySelector("#result");
const button = document.querySelector("#predict-button");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  button.firstChild.textContent = "Identifying...";
  result.hidden = true;

  const data = Object.fromEntries(new FormData(form));
  Object.keys(data).forEach((key) => { data[key] = Number(data[key]); });

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed. Please try again.");
    }
    result.className = "result success";
    result.innerHTML = `This iris is most likely <strong>${payload.Result}</strong>.`;
  } catch (error) {
    result.className = "result error";
    result.textContent = error.message;
  } finally {
    result.hidden = false;
    button.disabled = false;
    button.firstChild.textContent = "Identify species ";
  }
});
