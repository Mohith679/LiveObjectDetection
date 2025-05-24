const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const detectionOutput = document.getElementById("output");

// Get user media
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((err) => {
        console.error("Error accessing webcam:", err);
    });

video.addEventListener("loadedmetadata", () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Start capturing and sending frames
    setInterval(captureAndSendFrame, 1000);
});

function captureAndSendFrame() {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        fetch("http://localhost:8000/predict", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                displayDetections(data.detections);
            })
            .catch((err) => {
                console.error("Prediction error:", err);
            });
    }, "image/jpeg");
}

function displayDetections(detections) {
    detectionOutput.innerHTML = "";
    if (detections.length === 0) {
        detectionOutput.innerHTML = "No objects detected.";
    } else {
        detections.forEach((det) => {
            const p = document.createElement("p");
            p.textContent = `${det.label} (${(det.confidence * 100).toFixed(1)}%)`;
            detectionOutput.appendChild(p);
        });
    }
}
