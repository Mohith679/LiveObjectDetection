const detectBtn = document.getElementById("detectBtn");
const resultsDiv = document.getElementById("results");

detectBtn.addEventListener("click", async () => {
  resultsDiv.innerHTML = "Detecting...";
  try {
    const response = await fetch("http://localhost:8000/detect");
    const data = await response.json();
    
    if (data.detections) {
      resultsDiv.innerHTML = data.detections.map(d =>
        `<p><strong>${d.object}</strong>: ${d.distance_cm} cm - ${d.status}</p>`
      ).join("");
    } else {
      resultsDiv.innerHTML = `<p>${data.message || data.error}</p>`;
    }
  } catch (err) {
    resultsDiv.innerHTML = "Error calling API.";
  }
});
