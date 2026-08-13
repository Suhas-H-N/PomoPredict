// Webcam capture -> POST /predict_frame -> render the result inline.
// Lives in static/uploads/ alongside style.css since that's the app's one
// static folder; nothing here is actually a user upload.

(function () {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const captureBtn = document.getElementById("capture-btn");
  const statusEl = document.getElementById("status");
  const resultContainer = document.getElementById("result-container");

  if (!video || !captureBtn) return; // only run on the webcam page

  let stream = null;

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      video.srcObject = stream;
    } catch (err) {
      statusEl.textContent =
        "Couldn't access the camera (permission denied or none available). " +
        "You can still use the regular upload form.";
    }
  }

  function getLocation() {
    return new Promise((resolve) => {
      if (!navigator.geolocation) return resolve({});
      navigator.geolocation.getCurrentPosition(
        (pos) => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
        () => resolve({}),
        { timeout: 2000 }
      );
    });
  }

  function renderResult(r) {
    const pct = (n) => Math.round(n * 1000) / 10;

    const gradcam = r.gradcam_data_url
      ? `<details class="gradcam-toggle">
           <summary>Show Grad-CAM "why" heatmap</summary>
           <img src="${r.gradcam_data_url}" alt="Grad-CAM heatmap" class="preview">
         </details>`
      : "";

    const severity = r.severity
      ? `<p class="severity severity-${r.severity.toLowerCase()}">Estimated severity: ${r.severity}</p>`
      : "";

    const uncertain = r.is_uncertain
      ? `<p class="warn">Confidence is low — try a clearer, closer photo of the leaf/fruit.</p>`
      : "";

    const treatment = r.treatment
      ? `<p class="treatment"><strong>Suggested treatment:</strong> ${r.treatment}</p>`
      : "";

    const rows = r.raw
      .map(
        ([label, conf]) => `
        <li>
          <div class="pred-row">
            <span class="label">${label.replace(/_/g, " ")}</span>
            <span class="confidence">${pct(conf)}%</span>
          </div>
          <div class="bar-track"><div class="bar-fill" style="width:${pct(conf)}%"></div></div>
        </li>`
      )
      .join("");

    resultContainer.innerHTML = `
      <div class="result-card">
        <h2>${r.top_label.replace(/_/g, " ")}
          <span class="confidence-pill">${pct(r.top_confidence)}%</span>
        </h2>
        ${uncertain}
        ${severity}
        ${gradcam}
        ${treatment}
        <h3>All predictions</h3>
        <ul class="predictions">${rows}</ul>
      </div>`;
  }

  captureBtn.addEventListener("click", async () => {
    if (!stream) {
      statusEl.textContent = "Camera isn't active.";
      return;
    }
    captureBtn.disabled = true;
    statusEl.textContent = "Capturing…";

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
      const loc = await getLocation();
      const form = new FormData();
      form.append("image", blob, "capture.png");
      if (loc.lat) form.append("lat", loc.lat);
      if (loc.lon) form.append("lon", loc.lon);

      statusEl.textContent = "Diagnosing…";
      try {
        const res = await fetch("/predict_frame", { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok) {
          statusEl.textContent = data.error || "Something went wrong.";
        } else {
          statusEl.textContent = "";
          renderResult(data);
        }
      } catch (err) {
        statusEl.textContent = "Network error — is the server running?";
      } finally {
        captureBtn.disabled = false;
      }
    }, "image/png");
  });

  startCamera();
  window.addEventListener("beforeunload", () => {
    if (stream) stream.getTracks().forEach((t) => t.stop());
  });
})();
