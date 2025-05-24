const video = document.getElementById("video");
const detectBtn = document.getElementById("detectBtn");
const resultsDiv = document.getElementById("results");

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(err => {
    console.error("Webcam error:", err);
  });

// Capture frame and send to API
detectBtn.addEventListener("click", async () => {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const imageBase64 = canvas.toDataURL("image/jpeg");

  resultsDiv.innerHTML = "Detecting...";
  try {
    const response = await fetch("https://your-render-backend.onrender.com/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageBase64 })
    });

    const data = await response.json();
    if (data.detections) {
      resultsDiv.innerHTML = data.detections.map(d =>
        `<p><strong>${d.object}</strong>: ${d.distance_cm} cm - ${d.status}</p>`
      ).join("");
    } else {
      resultsDiv.innerHTML = `<p>${data.message || data.error}</p>`;
    }
  } catch (error) {
    resultsDiv.innerHTML = "API error: " + error.message;
  }
});
