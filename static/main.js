function callInput(mode, source, textId) {
    if (source == "file") {
        document.getElementById("getFile").click();
    }
    if (source == "url") {
        $("#loaddiv").show();
        $("#content").hide();
        document.getElementById("urlInputId").value = textId;
    }
    document.getElementById("selectedModeId").value = mode;
    document.getElementById("selectedModeId").checked = true;
}

function postFile() {
    $("#content ").hide();
    $("#progress-bar-file1 ").show();

    var formdata = new FormData();
    formdata.append('getFile', $('#getFile')[0].files[0]);
    var request = new XMLHttpRequest();

    request.upload.addEventListener('progress', function (e) {
        var file1Size = $('#getFile')[0].files[0].size;

        if (e.loaded <= file1Size) {
            var percent = Math.round(e.loaded / file1Size * 100);
            $('#progress-bar-file1').width(percent + '%').html(percent + '%');
        }

        if (e.loaded == e.total) {
            $('#progress-bar-file1').width(100 + '%').html(100 + '%Show all YOLO objects ');
            $("#loaddiv").show();
            $("#content").hide();
            $("#progress-bar-file1").hide();
        }
    });

    request.open('post', '/');
    request.timeout = 45000;
    request.send(formdata);
}

// var figure = $(".video").hover(hoverVideo, hideVideo);

// function hoverVideo(e) {
//     $('video', this).get(0).play();
// }

// function hideVideo(e) {
//     $('video', this).get(0).pause();
//     $('video', this).get(0).currentTime = 0;
// }
