resetSettings();

let started = 0;
let working = 0;
let frameWidth = 0;
let frameHeight = 0;
let maxFrames = 0;
let videoResetCommand = false;
let videoStopCommand = false;
let modeResetCommand = false;
let urlSourceResetCommand = false;
let screenshotCommand = false;
let urlSource = "";
let mode = "";
let superresModel = "LAPSRN";
let esrganModel = "FALCOON";

function callVideoOrImage(value) {
    document.getElementById("selectedVideoOrImageId").value = value;
    document.getElementById("selectedVideoOrImageId").checked = true;
}

function sendStartCommand() {
    videoResetCommand = true;
}

function sendStopCommand() {
    videoStopCommand = true;
}

function sendModeChangeCommand() {
    modeResetCommand = true;
    mode = $("#modesId").val();
}

function sendModelChangeCommand() {
    modeResetCommand = true;
    superresModel = $("#modelsSuperresId").val();
}

function sendEsrganChangeCommand() {
    modeResetCommand = true;
    esrganModel = $("#modelsEsrganId").val();
}

function sendScreenshotCommand() {
    screenshotCommand = true;
}

function resetSettings() {
    // document.getElementById("positionId").value = 1
    console.log("RESET");
    document.getElementById("saturationId").value = 100;
    document.getElementById("cannyBlurId").value = 5;
    document.getElementById("cannyThres1Id").value = 71;
    document.getElementById("cannyThres2Id").value = 21;
    document.getElementById("contrastId").value = 100;
    document.getElementById("confidenceId").value = 20;
    document.getElementById("brightnessId").value = 0;
    document.getElementById("lineThicknessId").value = 1;
    document.getElementById("denoiseId").value = 7;
    document.getElementById("denoise2Id").value = 10;
    document.getElementById("rcnnSizeId").value = 10;
    document.getElementById("rcnnBlurId").value = 9;
    document.getElementById("sobelId").value = 3;
    document.getElementById("asciiSizeId").value = 4;
    document.getElementById("asciiIntervalId").value = 10;
    document.getElementById("asciiThicknessId").value = 1;
    document.getElementById("colorCountId").value = 0;
    document.getElementById("resizeId").value = 1;
    document.getElementById("saturationId").value = 100;
    document.getElementById("sharpenId").value = 0;
    document.getElementById("sharpenId2").value = 0;

    if (document.getElementById("modesId").value == "e") {       
        document.getElementById("denoise2Id").value = 50;
        document.getElementById("saturationId").value = 170;
        document.getElementById("contrastId").value = 120;
        document.getElementById("sharpenId").value = 4;
        document.getElementById("cannyThres1Id").value = 71;
        document.getElementById("cannyThres2Id").value = 1;
    }
    if (document.getElementById("modesId").value == "o") {
        document.getElementById("denoise2Id").value = 10;
        document.getElementById("sharpenId").value = 3;
        document.getElementById("sharpenId2").value = 0;
    }
    if (document.getElementById("modesId").value == "k") {
        document.getElementById("rcnnSizeId").value = 20;
        document.getElementById("rcnnBlurId").value = 1;

    }
    if (document.getElementById("modesId").value == "r") {
        document.getElementById("denoise2Id").value = 30;
        document.getElementById("cannyBlurId").value = 5;
        document.getElementById("cannyThres1Id").value = 71;
        document.getElementById("cannyThres2Id").value = 1;
    }
    if (document.getElementById("modesId").value == "s") {
        document.getElementById("denoise2Id").value = 25;
    }

    if (document.getElementById("modesId").value == "q") {
        document.getElementById("contrastId").value = 200;
        document.getElementById("saturationId").value = 200;
        document.getElementById("rcnnBlurId").value = 1;
    }

    if (document.getElementById("modesId").value == "j") {
        document.getElementById("contrastId").value = 130;
        document.getElementById("saturationId").value = 150;
        document.getElementById("rcnnBlurId").value = 3;
        document.getElementById("cannyBlurId").value = 5;
        document.getElementById("lineThicknessId").value = 2;
        document.getElementById("denoise2Id").value = 50;
    }

     if (document.getElementById("modesId").value == "i") {
        document.getElementById("contrastId").value = 130;
        document.getElementById("saturationId").value = 150;
        document.getElementById("rcnnBlurId").value = 3;
        document.getElementById("cannyBlurId").value = 5;
        document.getElementById("lineThicknessId").value = 2;
    }
}

var toggleSlidersTimer = setInterval(function () {
    $.ajax({
        url: '/stats',
        type: 'POST',
        success: function (response) {
            console.log(response);
            //$("#num").html(response["value"]);
            $("#num").html("FRAME: " + response["value"]);
            $("#framesCount").html("TOTAL: " + response["totalFrames"]);
            $("#progress").html("POS: " + response["progress"] + "%");
            $("#fps").html("FPS: " + response["fps"]);
            $("#cpu").html("CPU: " + response["cpuUsage"] + "%");
            $("#freeRam").html("RAM: " + response["freeRam"] + "GB");
            frameWidth = response["frameWidth"]
            frameHeight = response["frameHeight"]
            maxFrames = response["totalFrames"]
            currentMode = response["currentMode"]

            if (response["screenshotReady"]) {
                window.open(response["screenshotPath"], '_blank');
            }

            document.getElementById("modesId").value = currentMode;
            if (started == 0) {
                resetSettings();
                started = 1;
            }
            // userTime = response["userTime"]
            // $("#frameSize").html(userTime);

            $("#frameSize").html(frameWidth + "x" + frameHeight);
            $("#positionId")
                .prop({
                    min: 1,
                    max: maxFrames - 1
                })

            working = response["workingOn"];

            // hideSliders();

            // SHOW ALL YOLO OBJECTS
            if (currentMode == "a") {
                $("#cannyBlurIdBlock").hide()
                $("#lineThicknessIdBlock").hide()
                $("#sharpenIdBlock").hide()
                $("#sharpenIdBlock2").hide()
                $("#denoiseIdBlock").hide()
                $("#denoise2IdBlock").hide()
                $("#rcnnSizeIdBlock").hide()
                $("#rcnnBlurIdBlock").hide()
                $("#sobelIdBlock").hide()
                $("#asciiSizeIdBlock").hide()
                $("#asciiIntervalIdBlock").hide()
                $("#asciiThicknessIdBlock").hide()
                $("#resizeIdBlock").hide()
                $("#colorCountIdBlock").hide()
                $("#superresIdBlock").hide()
                $("#esrganIdBlock").hide()
                $("#confidenceIdBlock").show()
                $("#cannyThres1IdBlock").hide()
                $("#cannyThres2IdBlock").hide()

                $("#downloadObjectsId").show()
            }
            // BLUR OBJECTS WITH ASCII
            if (currentMode == "b") {
                $("#cannyBlurIdBlock").hide()
                $("#lineThicknessIdBlock").hide()
                $("#sharpenIdBlock").hide()
                $("#sharpenIdBlock2").hide()
                $("#denoiseIdBlock").hide()
                $("#denoise2IdBlock").hide()
                $("#rcnnSizeIdBlock").hide()
                $("#rcnnBlurIdBlock").show()
                $("#sobelIdBlock").hide()
                $("#resizeIdBlock").hide()
                $("#downloadObjectsId").hide()
                $("#colorCountIdBlock").hide()
                $("#superresIdBlock").hide()
                $("#esrganIdBlock").hide()
                $("#confidenceIdBlock").show()
                $("#asciiSizeIdBlock").show()
                $("#asciiIntervalIdBlock").show()
                $("#asciiThicknessIdBlock").show()
                $("#cannyThres1IdBlock").hide()
                $("#cannyThres2IdBlock").hide()
            }
            // CANNY AND GAUSSIAN BLUR COLOR
            if (currentMode == "j") {
                $("#cannyBlurIdBlock").show()
                $("#lineThicknessIdBlock").show()
                $("#sharpenIdBlock").hide()
                $("#sharpenIdBlock2").hide()
                $("#denoiseIdBlock").show()
                $("#denoise2IdBlock").show()
                $("#rcnnSizeIdBlock").hide()
                $("#sobelIdBlock").hide()
                $("#confidenceIdBlock").show()
                $("#rcnnBlurIdBlock").show()
                $("#asciiSizeIdBlock").hide()
                $("#superresIdBlock").hide()
                $("#esrganIdBlock").hide()
                $("#asciiIntervalIdBlock").hide()
                $("#asciiThicknessIdBlock").hide()
                $("#resizeIdBlock").hide()
                $("#colorCountIdBlock").hide()
                $("#downloadObjectsId").hide()
                $("#cannyThres1IdBlock").show()
                $("#cannyThres2IdBlock").show()
            }
            // CANNY + ANIMATE BACKGROUND
            if (currentMode == "i") {
                $("#cannyBlurIdBlock").show()
                $("#lineThicknessIdBlock").show()
                $("#sharpenIdBlock").hide()
                $("#sharpenIdBlock2").hide()
                $("#denoiseIdBlock").show()
                $("#denoise2IdBlock").show()
                $("#rcnnSizeIdBlock").hide()
                $("#rcnnBlurIdBlock").hide()
                $("#sobelIdBlock").hide()
                $("#confidenceIdBlock").show()
                $("#asciiSizeIdBlock").hide()
                $("#asciiIntervalIdBlock").hide()
                $("#asciiThicknessIdBlock").hide()
                $("#resizeIdBlock").hide()
                $("#colorCountIdBlock").hide()
                $("#downloadObjectsId").hide()
                $("#superresIdBlock").hide()
                $("#esrganIdBlock").hide()
                $("#cannyThres1IdBlock").show()
                $("#cannyThres2IdBlock").show()
            }
            // GAUSSIAN BLUR BACKGROUND
            if (currentMode == "m") {
                $("#cannyBlurIdBlock").hide()
                $("#lineThicknessIdBlock").hide()
                $("#denoiseIdBlock").hide()
                $("#sharpenIdBlock").hide()
                $("#sharpenIdBlock2").hide()
                $("#denoise2IdBlock").hide()
                $("#sobelIdBlock").hide()
                $("#confidenceIdBlock").show()
                $("#rcnnSizeIdBlock").show()
                $("#rcnnBlurIdBlock").show()
                $("#asciiSizeIdBlock").hide()
                $("#asciiIntervalIdBlock").hide()
                $("#asciiThicknessIdBlock").hide()
                $("#resizeIdBlock").hide()
                $("#colorCountIdBlock").hide()
                $("#downloadObjectsId").hide()
                $("#superresIdBlock").hide()
                $("#esrganIdBlock").hide()
                $("#cannyThres1IdBlock").hide()
                $("#cannyThres2IdBlock").hide()
            }
            // FAKE COLORIZER
            if (currentMode == "k") {
                $("#cannyBlurIdBlock").hide()
                $("#lineThicknessIdBlock").hide()
                $("#sharpenIdBlock").hide()
                $("#sharpenIdBlock2").hide()
                $("#denoiseIdBlock").hide()
                $("#denoise2IdBlock").hide()
                $("#rcnnBlurIdBlock").show()
                $("#sobelIdBlock").hide()
                $("#rcnnSizeIdBlock").show()
                $("#confidenceIdBlock").show()
                $("#asciiSizeIdBlock").hide()
                $("#asciiIntervalIdBlock").hide()
                $("#asciiThicknessIdBlock").hide()
                $("#resizeIdBlock").hide()
                $("#colorCountIdBlock").hide()
                $("#downloadObjectsId").hide()
                $("#superresIdBlock").hide()
                $("#esrganIdBlock").hide()
                $("#cannyThres1IdBlock").hide()
                $("#cannyThres2IdBlock").hide()
            }
            // CAFFE NEURAL NETWORK COLORIZER
            if (currentMode == "f") {
                $("#cannyBlurIdBlock").hide()
                $("#confidenceIdBlock").hide()
                $("#lineThicknessIdBlock").hide()
                $("#sharpenIdBlock").hide()
                $("#sharpenIdBlock2").hide()
                $("#denoiseIdBlock").hide()
                $("#denoise2IdBlock").hide()
                $("#rcnnSizeIdBlock").hide()
                $("#rcnnBlurIdBlock").hide()
                $("#sobelIdBlock").hide()
                $("#asciiSizeIdBlock").hide()
                $("#asciiIntervalIdBlock").hide()
                $("#asciiThicknessIdBlock").hide()
                $("#resizeIdBlock").hide()
                $("#colorCountIdBlock").hide()
                $("#downloadObjectsId").hide()
                $("#superresIdBlock").hide()
                $("#esrganIdBlock").hide()
                $("#cannyThres1IdBlock").hide()
                $("#cannyThres2IdBlock").hide()
            }
            // CARTOON STYLE
            if (currentMode == "e") {
                $("#cannyBlurIdBlock").show()
                $("#lineThicknessIdBlock").show()
                $("#colorCountIdBlock").show()
                $("#sharpenIdBlock").show()
                $("#sharpenIdBlock2").show()
                $("#denoiseIdBlock").show()
                $("#denoise2IdBlock").show()
                $("#confidenceIdBlock").hide()
                $("#rcnnSizeIdBlock").hide()
                $("#rcnnBlurIdBlock").hide()
                $("#sobelIdBlock").hide()
                $("#asciiSizeIdBlock").hide()
                $("#asciiIntervalIdBlock").hide()
                $("#asciiThicknessIdBlock").hide()
                $("#resizeIdBlock").hide()
                $("#downloadObjectsId").hide()
                $("#superresIdBlock").hide()
                $("#esrganIdBlock").hide()
                $("#cannyThres1IdBlock").show()
                $("#cannyThres2IdBlock").show()
            }
            // PENSIL DRAWER
            if (currentMode == "r") {
                $("#cannyBlurIdBlock").show()
                $("#lineThicknessIdBlock").show()
                $("#colorCountIdBlock").hide()
                $("#sharpenIdBlock").show()
                $("#sharpenIdBlock2").show()
                $("#denoiseIdBlock").show()
                $("#denoise2IdBlock").show()
                $("#confidenceIdBlock").hide()
                $("#rcnnSizeIdBlock").hide()
                $("#rcnnBlurIdBlock").hide()
                $("#sobelIdBlock").hide()
                $("#asciiSizeIdBlock").hide()
                $("#asciiIntervalIdBlock").hide()
                $("#asciiThicknessIdBlock").hide()
                $("#resizeIdBlock").hide()
                $("#downloadObjectsId").hide()
                $("#superresIdBlock").hide()
                $("#esrganIdBlock").hide()
                $("#cannyThres1IdBlock").show()
                $("#cannyThres2IdBlock").show()
            }
            // TWO-COLORED
            if (currentMode == "s") {
                $("#cannyBlurIdBlock").hide()
                $("#lineThicknessIdBlock").hide()
                $("#colorCountIdBlock").hide()
                $("#sharpenIdBlock").show()
                $("#sharpenIdBlock2").show()
                $("#denoiseIdBlock").show()
                $("#denoise2IdBlock").show()
                $("#confidenceIdBlock").hide()
                $("#rcnnSizeIdBlock").hide()
                $("#rcnnBlurIdBlock").hide()
                $("#sobelIdBlock").hide()
                $("#asciiSizeIdBlock").hide()
                $("#asciiIntervalIdBlock").hide()
                $("#asciiThicknessIdBlock").hide()
                $("#resizeIdBlock").hide()
                $("#downloadObjectsId").hide()
                $("#superresIdBlock").hide()
                $("#esrganIdBlock").hide()
                $("#cannyThres1IdBlock").hide()
                $("#cannyThres2IdBlock").hide()
            }
            // DENOISE + SHARPEN
            if (currentMode == "o") {
                $("#cannyBlurIdBlock").hide()
                $("#lineThicknessIdBlock").hide()
                $("#confidenceIdBlock").hide()
                $("#rcnnSizeIdBlock").hide()
                $("#rcnnBlurIdBlock").hide()
                $("#sobelIdBlock").hide()
                $("#asciiSizeIdBlock").hide()
                $("#asciiIntervalIdBlock").hide()
                $("#asciiThicknessIdBlock").hide()
                $("#resizeIdBlock").hide()
                $("#colorCountIdBlock").hide()
                $("#downloadObjectsId").hide()
                $("#superresIdBlock").hide()
                $("#esrganIdBlock").hide()
                $("#sharpenIdBlock").show()
                $("#sharpenIdBlock2").show()
                $("#denoiseIdBlock").show()
                $("#denoise2IdBlock").show()
                $("#cannyThres1IdBlock").hide()
                $("#cannyThres2IdBlock").hide()
            }
            // SOBEL OPERATOR
            if (currentMode == "p") {
                $("#cannyBlurIdBlock").hide()
                $("#lineThicknessIdBlock").hide()
                $("#confidenceIdBlock").hide()
                $("#rcnnSizeIdBlock").hide()
                $("#rcnnBlurIdBlock").hide()
                $("#asciiSizeIdBlock").hide()
                $("#asciiIntervalIdBlock").hide()
                $("#asciiThicknessIdBlock").hide()
                $("#resizeIdBlock").hide()
                $("#colorCountIdBlock").hide()
                $("#downloadObjectsId").hide()
                $("#superresIdBlock").hide()
                $("#esrganIdBlock").hide()
                $("#sharpenIdBlock").show()
                $("#sharpenIdBlock2").show()
                $("#denoiseIdBlock").show()
                $("#denoise2IdBlock").show()
                $("#sobelIdBlock").show()
                $("#cannyThres1IdBlock").hide()
                $("#cannyThres2IdBlock").hide()
            }
            // ASCII FULL SCREEN
            if (currentMode == "q") {
                $("#cannyBlurIdBlock").hide()
                $("#lineThicknessIdBlock").hide()
                $("#confidenceIdBlock").hide()
                $("#rcnnSizeIdBlock").hide()
                $("#rcnnBlurIdBlock").show()
                $("#sharpenIdBlock").hide()
                $("#sharpenIdBlock2").hide()
                $("#denoiseIdBlock").hide()
                $("#denoise2IdBlock").hide()
                $("#sobelIdBlock").hide()
                $("#resizeIdBlock").hide()
                $("#colorCountIdBlock").hide()
                $("#downloadObjectsId").hide()
                $("#superresIdBlock").hide()
                $("#esrganIdBlock").hide()
                $("#asciiSizeIdBlock").show()
                $("#asciiIntervalIdBlock").show()
                $("#asciiThicknessIdBlock").show()
                $("#cannyThres1IdBlock").hide()
                $("#cannyThres2IdBlock").hide()
            }
            // UPSCALER
            if (currentMode == "n") {
                $("#cannyBlurIdBlock").hide()
                $("#lineThicknessIdBlock").hide()
                $("#confidenceIdBlock").hide()
                $("#rcnnSizeIdBlock").hide()
                $("#rcnnBlurIdBlock").hide()
                $("#sharpenIdBlock").show()
                $("#sharpenIdBlock2").show()
                $("#denoiseIdBlock").hide()
                $("#colorCountIdBlock").hide()
                $("#denoise2IdBlock").hide()
                $("#sobelIdBlock").hide()
                $("#asciiSizeIdBlock").hide()
                $("#asciiIntervalIdBlock").hide()
                $("#asciiThicknessIdBlock").hide()
                $("#downloadObjectsId").hide()
                $("#resizeIdBlock").hide()
                $("#cannyThres1IdBlock").hide()
                $("#cannyThres2IdBlock").hide()

                $("#superresIdBlock").show()
                $("#esrganIdBlock").hide()
            }
            // ESRGAN UPSCALER
            if (currentMode == "t") {
                $("#cannyBlurIdBlock").hide()
                $("#lineThicknessIdBlock").hide()
                $("#confidenceIdBlock").hide()
                $("#rcnnSizeIdBlock").hide()
                $("#rcnnBlurIdBlock").hide()
                $("#sharpenIdBlock").show()
                $("#sharpenIdBlock2").show()
                $("#denoiseIdBlock").hide()
                $("#colorCountIdBlock").hide()
                $("#denoise2IdBlock").hide()
                $("#sobelIdBlock").hide()
                $("#asciiSizeIdBlock").hide()
                $("#asciiIntervalIdBlock").hide()
                $("#asciiThicknessIdBlock").hide()
                $("#resizeIdBlock").hide()
                $("#downloadObjectsId").hide()
                $("#cannyThres1IdBlock").hide()
                $("#cannyThres2IdBlock").hide()
                $("#superresIdBlock").hide()
                $("#esrganIdBlock").show()
            }

            // ESRGAN UPSCALER
            if (currentMode == "z") {
                $("#cannyBlurIdBlock").hide()
                $("#lineThicknessIdBlock").hide()
                $("#confidenceIdBlock").hide()
                $("#rcnnSizeIdBlock").hide()
                $("#rcnnBlurIdBlock").hide()
                $("#sharpenIdBlock").hide()
                $("#sharpenIdBlock2").hide()
                $("#denoiseIdBlock").hide()
                $("#colorCountIdBlock").hide()
                $("#denoise2IdBlock").hide()
                $("#sobelIdBlock").hide()
                $("#asciiSizeIdBlock").hide()
                $("#asciiIntervalIdBlock").hide()
                $("#asciiThicknessIdBlock").hide()
                $("#resizeIdBlock").hide()
                $("#downloadObjectsId").hide()
                $("#cannyThres1IdBlock").hide()
                $("#cannyThres2IdBlock").hide()
                $("#superresIdBlock").hide()
                $("#esrganIdBlock").hide()

            }
        },
        error: function (error) {
            console.log(error);
        }
    })
}, 1000);

let sendSettingsTimer = setInterval(function () {
    let viewSource = false;
    if (document.getElementById("viewSourceId").checked) {
        viewSource = true;
    }
    else {
        viewSource = false;
    }

    let positionSlider = document.getElementById("positionId");
    let outputPosition = document.getElementById("positionValue");
    let positionSliderValue = positionSlider.value;

    let saturationSlider = document.getElementById("saturationId");
    let saturationOutput = document.getElementById("saturationValue");
    let saturationSliderValue = saturationSlider.value;

    let cannyBlurSlider = document.getElementById("cannyBlurId");
    let cannyBlurOutput = document.getElementById("cannyBlurValue");
    let cannyBlurSliderValue = cannyBlurSlider.value;

    let cannyThres1Slider = document.getElementById("cannyThres1Id");
    let cannyThres1Output = document.getElementById("cannyThres1Value");
    let cannyThresSliderValue = cannyThres1Slider.value;

    let cannyThres2Slider = document.getElementById("cannyThres2Id");
    let cannyThres2Output = document.getElementById("cannyThres2Value");
    let cannyThresSliderValue2 = cannyThres2Slider.value;

    let contrastSlider = document.getElementById("contrastId");
    let contrastOutput = document.getElementById("contrastValue");
    let contrastSliderValue = contrastSlider.value;

    let confidenceSlider = document.getElementById("confidenceId");
    let confidenceOutput = document.getElementById("confidenceValue");
    let confidenceSliderValue = confidenceSlider.value;

    let brightnessSlider = document.getElementById("brightnessId");
    let brightnessOutput = document.getElementById("brightnessValue");
    let brightnessSliderValue = brightnessSlider.value;

    let lineThicknessSlider = document.getElementById("lineThicknessId");
    let lineThicknessOutput = document.getElementById("lineThicknessValue");
    let lineThicknessSliderValue = lineThicknessSlider.value;

    let denoiseSlider = document.getElementById("denoiseId");
    let denoiseOutput = document.getElementById("denoiseValue");
    let denoiseSliderValue = denoiseSlider.value;

    let denoise2Slider = document.getElementById("denoise2Id");
    let denoise2Output = document.getElementById("denoise2Value");
    let denoiseSliderValue2 = denoise2Slider.value;

    let sharpenSlider = document.getElementById("sharpenId");
    let sharpenOutput = document.getElementById("sharpenValue");
    let sharpenSliderValue = sharpenSlider.value;

    let sharpenSlider2 = document.getElementById("sharpenId2");
    let sharpenOutput2 = document.getElementById("sharpenValue2");
    let sharpenSliderValue2 = sharpenSlider2.value;

    let rcnnSizeSlider = document.getElementById("rcnnSizeId");
    let rcnnSizeOutput = document.getElementById("rcnnSizeValue");
    let rcnnSizeSliderValue = rcnnSizeSlider.value;

    let rcnnBlurSlider = document.getElementById("rcnnBlurId");
    let rcnnBlurOutput = document.getElementById("rcnnBlurValue");
    let rcnnBlurSliderValue = rcnnBlurSlider.value;

    let sobelSlider = document.getElementById("sobelId");
    let sobelOutput = document.getElementById("sobelValue");
    let sobelSliderValue = sobelSlider.value;

    let asciiSizeSlider = document.getElementById("asciiSizeId");
    let asciiSizeOutput = document.getElementById("asciiSizeValue");
    let asciiSizeSliderValue = asciiSizeSlider.value;

    let asciiIntervalSlider = document.getElementById("asciiIntervalId");
    let asciiIntervalOutput = document.getElementById("asciiIntervalValue");
    let asciiIntervalSliderValue = asciiIntervalSlider.value;

    let asciiThicknessSlider = document.getElementById("asciiThicknessId");
    let asciiThicknessOutput = document.getElementById("asciiThicknessValue");
    let asciiThicknessSliderValue = asciiThicknessSlider.value;

    let resizeSlider = document.getElementById("resizeId");
    let resizeOutput = document.getElementById("resizeValue");
    let resizeSliderValue = resizeSlider.value;

    let colorCountSlider = document.getElementById("colorCountId");
    let colorCountOutput = document.getElementById("colorCountValue");
    let colorCountSliderValue = colorCountSlider.value;

    outputPosition.innerHTML = positionSlider.value;
    saturationOutput.innerHTML = saturationSlider.value;
    cannyBlurOutput.innerHTML = cannyBlurSlider.value;
    cannyThres1Output.innerHTML = cannyThres1Slider.value;
    cannyThres2Output.innerHTML = cannyThres2Slider.value;
    contrastOutput.innerHTML = contrastSlider.value;
    brightnessOutput.innerHTML = brightnessSlider.value;
    confidenceOutput.innerHTML = confidenceSlider.value;
    lineThicknessOutput.innerHTML = lineThicknessSlider.value;
    denoiseOutput.innerHTML = denoiseSlider.value;
    denoise2Output.innerHTML = denoise2Slider.value;
    sharpenOutput.innerHTML = sharpenSlider.value;
    sharpenOutput2.innerHTML = sharpenSlider2.value;
    rcnnSizeOutput.innerHTML = rcnnSizeSlider.value;
    rcnnBlurOutput.innerHTML = rcnnBlurSlider.value;
    sobelOutput.innerHTML = sobelSlider.value;
    asciiSizeOutput.innerHTML = asciiSizeSlider.value;
    asciiIntervalOutput.innerHTML = asciiIntervalSlider.value;
    asciiThicknessOutput.innerHTML = asciiThicknessSlider.value;
    resizeOutput.innerHTML = resizeSlider.value;
    colorCountOutput.innerHTML = colorCountSlider.value;

    positionSlider.oninput = function () {
        outputPosition.innerHTML = this.value;
    }

    saturationSlider.oninput = function () {
        saturationOutput.innerHTML = this.value;
    }

    cannyBlurSlider.oninput = function () {
        cannyBlurOutput.innerHTML = this.value;
    }

    cannyThres1Slider.oninput = function () {
        cannyThres1Output.innerHTML = this.value;
    }

    cannyThres2Slider.oninput = function () {
        cannyThres2Output.innerHTML = this.value;
    }

    contrastSlider.oninput = function () {
        contrastOutput.innerHTML = this.value;
    }

    brightnessSlider.oninput = function () {
        brightnessOutput.innerHTML = this.value;
    }

    confidenceSlider.oninput = function () {
        confidenceOutput.innerHTML = this.value;
    }

    lineThicknessSlider.oninput = function () {
        lineThicknessOutput.innerHTML = this.value;
    }

    denoiseSlider.oninput = function () {
        denoiseOutput.innerHTML = this.value;
    }

    denoise2Slider.oninput = function () {
        denoise2Output.innerHTML = this.value;
    }

    sharpenSlider.oninput = function () {
        sharpenOutput.innerHTML = this.value;
    }

    sharpenSlider2.oninput = function () {
        sharpenOutput2.innerHTML = this.value;
    }

    rcnnSizeSlider.oninput = function () {
        rcnnSizeOutput.innerHTML = this.value;
    }

    rcnnBlurSlider.oninput = function () {
        rcnnBlurOutput.innerHTML = this.value;
    }

    sobelSlider.oninput = function () {
        sobelOutput.innerHTML = this.value;
    }

    asciiSizeSlider.oninput = function () {
        asciiSizeOutput.innerHTML = this.value;
    }

    asciiIntervalSlider.oninput = function () {
        asciiIntervalOutput.innerHTML = this.value;
    }

    asciiThicknessSlider.oninput = function () {
        asciiThicknessOutput.innerHTML = this.value;
    }

    resizeSlider.oninput = function () {
        resizeOutput.innerHTML = this.value;
    }

    colorCountSlider.oninput = function () {
        colorCountOutput.innerHTML = this.value;
    }

    $.ajax({
        type: "POST",
        contentType: "application/json;charset=utf-8",
        url: "/settings",
        traditional: "true",
        data: JSON.stringify({
            viewSource,
            cannyBlurSliderValue,
            cannyThresSliderValue,
            cannyThresSliderValue2,
            saturationSliderValue,
            contrastSliderValue,
            brightnessSliderValue,
            positionSliderValue,
            confidenceSliderValue,
            lineThicknessSliderValue,
            denoiseSliderValue,
            denoiseSliderValue2,
            sharpenSliderValue,
            sharpenSliderValue2,
            rcnnSizeSliderValue,
            rcnnBlurSliderValue,
            sobelSliderValue,
            asciiSizeSliderValue,
            asciiIntervalSliderValue,
            asciiThicknessSliderValue,
            resizeSliderValue,
            colorCountSliderValue,
            videoResetCommand,
            videoStopCommand,
            modeResetCommand,
            screenshotCommand,
            urlSourceResetCommand,
            urlSource,
            mode,
            superresModel,
            esrganModel
        }),
        dataType: "json"
    });

    screenshotCommand = false;
    videoResetCommand = false;
    videoStopCommand = false;
    modeResetCommand = false;
    superresResetCommand = false;
    estranResetCommand = false;
    urlSourceResetCommand = false;

}, 300);