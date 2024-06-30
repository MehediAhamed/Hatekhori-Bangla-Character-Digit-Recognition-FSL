
document.addEventListener("DOMContentLoaded", function() {
    const canvas = document.getElementById('main-canvas');
    const displayPrediction = document.getElementById('prediction');
    const displayPrompt = document.getElementById('prompt');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;

    const banglaCharacters = [
        { char: 'অ', audio: '/audio/character_0.mp3' },
        { char: 'আ', audio: '/audio/character_1.mp3' },
        { char: 'ই', audio: '/audio/character_2.mp3' },
        { char: 'ঈ', audio: '/audio/character_3.mp3' },
        { char: 'উ', audio: '/audio/character_4.mp3' },
        { char: 'ঊ', audio: '/audio/character_5.mp3' },
        { char: 'ঋ', audio: '/audio/character_6.mp3' },
        { char: 'এ', audio: '/audio/character_7.mp3' },
        { char: 'ঐ', audio: '/audio/character_8.mp3' },
        { char: 'ও', audio: '/audio/character_9.mp3' },
        { char: 'ঔ', audio: '/audio/character_10.mp3' },
        { char: 'ক', audio: '/audio/character_11.mp3' },
        { char: 'খ', audio: '/audio/character_12.mp3' },
        { char: 'গ', audio: '/audio/character_13.mp3' },
        { char: 'ঘ', audio: '/audio/character_14.mp3' },
        { char: 'ঙ', audio: '/audio/character_15.mp3' },
        { char: 'চ', audio: '/audio/character_16.mp3' },
        { char: 'ছ', audio: '/audio/character_17.mp3' },
        { char: 'জ', audio: '/audio/character_18.mp3' },
        { char: 'ঝ', audio: '/audio/character_19.mp3' },
        { char: 'ঞ', audio: '/audio/character_20.mp3' },
        { char: 'ট', audio: '/audio/character_21.mp3' },
        { char: 'ঠ', audio: '/audio/character_22.mp3' },
        { char: 'ড', audio: '/audio/character_23.mp3' },
        { char: 'ঢ', audio: '/audio/character_24.mp3' },
        { char: 'ণ', audio: '/audio/character_25.mp3' },
        { char: 'ত', audio: '/audio/character_26.mp3' },
        { char: 'থ', audio: '/audio/character_27.mp3' },
        { char: 'দ', audio: '/audio/character_28.mp3' },
        { char: 'ধ', audio: '/audio/character_29.mp3' },
        { char: 'ন', audio: '/audio/character_30.mp3' },
        { char: 'প', audio: '/audio/character_31.mp3' },
        { char: 'ফ', audio: '/audio/character_32.mp3' },
        { char: 'ব', audio: '/audio/character_33.mp3' },
        { char: 'ভ', audio: '/audio/character_34.mp3' },
        { char: 'ম', audio: '/audio/character_35.mp3' },
        { char: 'য', audio: '/audio/character_36.mp3' },
        { char: 'র', audio: '/audio/character_37.mp3' },
        { char: 'ল', audio: '/audio/character_38.mp3' },
        { char: 'শ', audio: '/audio/character_39.mp3' },
        { char: 'ষ', audio: '/audio/character_40.mp3' },
        { char: 'স', audio: '/audio/character_41.mp3' },
        { char: 'হ', audio: '/audio/character_42.mp3' },
        { char: 'ড়', audio: '/audio/character_43.mp3' },
        { char: 'ঢ়', audio: '/audio/character_44.mp3' },
        { char: 'য়', audio: '/audio/character_45.mp3' },
        { char: 'ৎ', audio: '/audio/character_46.mp3' },
        { char: '০', audio: '/audio/number_0.mp3' },
        { char: '১', audio: '/audio/number_1.mp3' },
        { char: '২', audio: '/audio/number_2.mp3' },
        { char: '৩', audio: '/audio/number_3.mp3' },
        { char: '৪', audio: '/audio/number_4.mp3' },
        { char: '৫', audio: '/audio/number_5.mp3' },
        { char: '৬', audio: '/audio/number_6.mp3' },
        { char: '৭', audio: '/audio/number_7.mp3' },
        { char: '৮', audio: '/audio/number_8.mp3' },
        { char: '৯', audio: '/audio/number_9.mp3' }
    ];


    // Randomly select a Bangla character and play its audio
    function playRandomCharacter() {
        const randomIndex = Math.floor(Math.random() * banglaCharacters.length);
        const randomCharacter = banglaCharacters[randomIndex];
        const audioElement = document.getElementById('bangla-audio');
        audioElement.src = randomCharacter.audio;
        audioElement.play();
        displayPrompt.innerText = `Random Character: ${randomCharacter.char}`;
        return randomCharacter;
    }

    // Play audio on button click
    function playAudio() {
        const audioElement = document.getElementById('bangla-audio');
        audioElement.play();
    }

    // Initialize canvas with appropriate settings
    function initializeCanvas() {
        canvas.width = window.innerWidth * 0.25;
        canvas.height = window.innerHeight * 0.5;
        ctx.fillStyle = 'black'; // Set the background to black
        ctx.fillRect(0, 0, canvas.width, canvas.height); // Fill the background
        ctx.strokeStyle = 'white'; // Set the drawing color to white
        ctx.lineWidth = 10; // Set the line width for drawing
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
    }

    // Get the correct mouse position relative to the canvas
    function getMousePosition(event) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
    }

    // Start drawing
    function startDrawing(e) {
        isDrawing = true;
        ctx.beginPath();
        const pos = getMousePosition(e);
        ctx.moveTo(pos.x, pos.y);
    }

    // Drawing
    function draw(e) {
        if (!isDrawing) return;
        const pos = getMousePosition(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }

    // Stop drawing
    function stopDrawing() {
        isDrawing = false;
        ctx.beginPath();
    }

    // Erase the canvas
    function eraseCanvas() {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        displayPrediction.innerText = 'Canvas cleared.';
    }

    // Send the canvas content to Flask backend for prediction
    function predict() {
        const dataURL = canvas.toDataURL('image/jpeg');
        
        // Convert the dataURL to a blob
        const byteString = atob(dataURL.split(',')[1]);
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        const blob = new Blob([ab], { type: 'image/jpeg' });
        
        const formData = new FormData();
        formData.append('file', blob, 'drawing.jpg');

        fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            console.log('Prediction:', data);
            if (data.extracted_text) {
                // Extract the first character from the predicted text
                const firstCharacter = data.extracted_text.trim().charAt(0);
                displayPrediction.innerText = `Predicted Character: ${firstCharacter}`;
                
               
            } else {
                displayPrediction.innerText = 'No text recognized.';
            }
        })
        .catch(error => {
            console.error('Prediction error:', error);
            displayPrediction.innerText = 'Error predicting character.';
        });
    }
    

    // Event listeners for drawing
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch events for drawing on touch devices
    canvas.addEventListener('touchstart', (event) => {
        isDrawing = true;
        const touch = event.touches[0];
        const pos = getMousePosition(touch);
        ctx.moveTo(pos.x, pos.y);
    });
    
    canvas.addEventListener('touchend', stopDrawing);
    
    canvas.addEventListener('touchmove', (event) => {
        if (!isDrawing) return;
        const touch = event.touches[0];
        const pos = getMousePosition(touch);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        event.preventDefault();
    });

    // Controls
    document.getElementById('erase').addEventListener('click', eraseCanvas);
    document.getElementById('predict-button').addEventListener('click', predict);
    document.getElementById('speaker-button').addEventListener('click', playAudio);

    // Adjust canvas on resize
    function resizeCanvas() {
        initializeCanvas();
    }
    window.addEventListener('resize', resizeCanvas);

    initializeCanvas(); // Initial canvas setup
    playRandomCharacter(); // Play a random character sound on page load
    

});
