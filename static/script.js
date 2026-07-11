let lastWord = "";

function fetchGesture() {
    fetch('/gesture_text')
        .then(response => response.text())
        .then(word => {
            if (word !== lastWord && word !== "..." && word !== "") {
                lastWord = word;
                document.getElementById("gesture-text").innerText = word;

                // Speak the word
                const utter = new SpeechSynthesisUtterance(word);
                window.speechSynthesis.speak(utter);
            }
        })
        .catch(err => console.error(err));
}

// Check every 500ms
setInterval(fetchGesture, 500);
