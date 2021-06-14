import * as wasm from "wasm-zipnet";

// wasm.greet();

/// Prompts user to download the passed data
/// https://stackoverflow.com/questions/23451726/saving-binary-data-as-file-using-javascript-from-a-browser
function promptDownload(data) {
    var saveByteArray = (function () {
        var a = document.createElement("a");
        document.body.appendChild(a);
        a.style = "display: none";
        return function (data, name) {
            var blob = new Blob(data, { type: "octet/stream" }),
                url = window.URL.createObjectURL(blob);
            a.href = url;
            a.download = name;
            a.click();
            window.URL.revokeObjectURL(url);
        };
    }());
    saveByteArray([data], "zipnet-enc-" + Date.now() + ".bin");
}

/// Displays a file selected by the user in a preview.
function previewImageToEncode() {
    var preview = document.getElementById("encodedPreview");
    var file = document.getElementById("imgInput").files[0];
    var reader = new FileReader();

    reader.onloadend = function () {
        preview.src = reader.result;
    }

    if (file) {
        reader.readAsDataURL(file);
    } else {
        preview.src = "";
    }
}

/// Starts the encoding process and then opens a download prompt for the user.
function downloadEncodedImage() {
    var file = document.getElementById("imgInput").files[0];
    var reader = new FileReader();

    reader.onloadend = function () {
        var data = new Uint8Array(reader.result);

        // Measuring exection time for the statistics
        var start = performance.now();
        var encoded = wasm.encode_image(data);
        var end = performance.now();

        // TODO: Use actual (and maybe more interesting) stats
        displayEncoderStats(data.length, encoded.length, 0, end - start);

        promptDownload(encoded);
    }

    // Fallout if no file is selected beforehand.
    if (file) {
        reader.readAsArrayBuffer(file)
    }
    else {
        alert("Please first select an image to encode.");
        console.log("Tried to encode without selected image.");
        return;
    }
}

function displayEncoderStats(prev_size, new_size, PSNR, time) {
    var stats_div = document.getElementById("statsEncoded").innerHTML = "<p> Previous size: "
        + prev_size + " Byte, Encoded size: " + new_size + " Byte, PSNR: " + PSNR + ", Encoding time: " + time + " ms</p>";
}

function displayDecoderStats(time) {
    var stats_div = document.getElementById("statsDecoded").innerHTML = "<p>Decoding time: " + time + " ms</p>";
}

/// Decodes image and displays it in the preview.
function decodeImageAndPreview() {
    var preview = document.getElementById("decodedPreview");
    var file = document.getElementById('encodedInput').files[0];
    var reader = new FileReader();

    reader.onloadend = function () {
        var data = new Uint8Array(reader.result);

        // Measuring time for the performance stats.
        var start = performance.now();
        var decoded = wasm.decode_image(data).buffer;
        var end = performance.now();

        displayDecoderStats(end - start);

        // Puts the decoded data into the preview (in a convoluted way)
        var decoded_blob = new Blob([decoded]);
        var preview_reader = new FileReader();
        preview_reader.onload = function (e) {
            preview.src = e.target.result;
        }
        preview_reader.readAsDataURL(decoded_blob);
    }

    if (file) {
        reader.readAsArrayBuffer(file)
    }
}

/// Prompts the user to download the decoded image that is shown in the decoder preview.
function downloadDecodedImage() {
    var img_dataurl = document.getElementById("decodedPreview").src;
    if (img_dataurl == "") {
        alert("Please first select an encoded image you want to decode.");
        console.log("Tried to download empty preview.");
        return;
    }

    var a = document.createElement("a");
    document.body.appendChild(a);
    a.style = "display: none";
    a.href = img_dataurl;
    a.download = "zipnet-decoded-" + Date.now() + ".png";
    a.click();
    window.URL.revokeObjectURL(img_dataurl);
}

document.getElementById('imgInput').addEventListener("change", previewImageToEncode);
document.getElementById("downloadEncoded").addEventListener("click", downloadEncodedImage);
document.getElementById('encodedInput').addEventListener("change", decodeImageAndPreview);
document.getElementById("downloadDecoded").addEventListener("click", downloadDecodedImage);
