const form = document.getElementById("predict-form");
const result = document.getElementById("result");
const scoreEl = document.getElementById("result-score");
const noteEl = document.getElementById("result-note");
const errorEl = document.getElementById("error");
const submitBtn = document.getElementById("submit-btn");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorEl.hidden = true;
  result.hidden = true;
  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting…";

  const data = Object.fromEntries(new FormData(form).entries());
  const numeric = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1",
  ];
  for (const key of numeric) data[key] = Number(data[key]);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = payload.detail;
      throw new Error(typeof detail === "string" ? detail : "Prediction failed.");
    }

    const passed = Boolean(payload.passed);
    const pct = payload.probability != null
      ? Math.round(payload.probability * 100)
      : null;
    scoreEl.textContent = payload.label || (passed ? "Pass" : "Fail");
    noteEl.textContent = pct == null
      ? (passed
        ? "The model expects this student to pass the course."
        : "The model expects this student to fail the course.")
      : (passed
        ? `The model estimates a ${pct}% chance this student will pass.`
        : `The model estimates a ${pct}% chance this student will fail.`);
    result.classList.toggle("pass", passed);
    result.classList.toggle("fail", !passed);
    result.hidden = false;
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict outcome";
  }
});
