function uploadImage() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image!");
        return;
    }

    const preview = document.getElementById("preview");
    const result = document.getElementById("result");
    const bar = document.getElementById("bar");

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";

    result.innerText = "Predicting...";
    bar.style.width = "0%";
    bar.style.background = "green";

    let progress = 0;
    let interval = setInterval(() => {
        if (progress < 70) {
            progress += 5;
            bar.style.width = progress + "%";
        } else {
            clearInterval(interval);
        }
    }, 200);

    const formData = new FormData();
    formData.append("file", file);

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        clearInterval(interval);

        if (data.error) {
            result.innerText = "Error: " + data.error;
            bar.style.width = "0%";
            return;
        }

        const formatted = data.prediction
            .replaceAll("_", " ")
            .replaceAll("-", " ");

        const confidence = Number(data.confidence);

        if (confidence < 60) {
            result.innerText = `⚠️ Low confidence: ${formatted} (${confidence.toFixed(2)}%)`;
        } else {
            result.innerText = `Prediction: ${formatted} (${confidence.toFixed(2)}% confident)`;
        }

        bar.style.width = confidence + "%";

        if (confidence > 90) {
            bar.style.background = "green";
        } else if (confidence > 50) {
            bar.style.background = "orange";
        } else {
            bar.style.background = "red";
        }
    })
    .catch(err => {
        clearInterval(interval);
        console.error(err);
        result.innerText = "Error connecting to backend";
        bar.style.width = "0%";
    });
}