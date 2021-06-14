import * as wasm from "wasm-zipnet";

// wasm.greet();

// Displays a file selected by the user in a preview.
function previewFile() {
    var preview = document.querySelector('img');
    var file = document.querySelector('input[type=file]').files[0];
    var reader = new FileReader();

    reader.onloadend = function () {
        // preview.src = reader.result;

        var data = reader.result;
        var len = wasm.process_png_image(new Uint8Array(data));
        alert(len);
    }

    if (file) {
        reader.readAsArrayBuffer(file)

        // reader.readAsDataURL(file);
    } else {
        preview.src = "";
    }
}
document.querySelector('input').addEventListener("change", previewFile);
