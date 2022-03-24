import { Imag, image } from "@tensorflow/tfjs";
import * as wasm from "wasm-zipnet";
const { Image } = require("image-js");

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

var last_bpp = undefined;

/// Displays a file selected by the user in a preview.
function previewImageToEncode() {
    var preview = document.getElementById("encodedPreview");
    var file = document.getElementById("imgInput").files[0];

    if (file) {
        file.arrayBuffer().then(
            (buffer) => displayBuffer(preview, buffer)
        )
    } else {
        preview.src = "";
    }
}

/// Encodes the given image with the given JPEG quality.
function imgToJpeg(quality, image) {
    return image.toBuffer({
        format: "jpeg",
        encoder: {
            quality: quality
        }
    })
}

/// Finds the closest bpp to the target.
/// We expect an image-js Image.
function findBpp(target_bpp, image) {
    var n_pixels = getNumPixels(image);
    var high = 100;
    var low = 0;
    var n_it = 0; // failswitch
    while (high != low && n_it < 20) {
        n_it += 1;
        var current = Math.round((high + low) / 2);
        var buffer = imgToJpeg(current, image);
        var bits = 8 * buffer.length;
        var bpp = bits / n_pixels;
        if (bpp > target_bpp) {
            high = current - 1;
        }
        else if (bpp < target_bpp) {
            low = current;
        }
    }
    // So we are definitely over the target bpp, wouldn't want to be unfair
    // by giving JPEG less bpp.
    if (bpp < target_bpp) {
        current += 1;
    }
    return { buffer: buffer, quality: current, bpp: bpp };
}

/// Call back for failing to read an image.
function imageReadFail(error) {
    alert("Could not read image: " + error);
}

/// Returns the number of pixel at the given Path. Path
/// has to be loadable by the Image.load function.
/// Returns number of pixel in the image.
/// A pixel is the same in RGB and grayscale images.
function getNumPixels(image) {
    return image.width * image.height;
}

function displayJpegStats(bpp, psnr) {
    document.getElementById("statsJpeg").innerHTML = "JPEG: " + bpp + " bpp, " + " PSNR";
}

function calculatePsnr(image1, image2) {
    var p1 = image1.getPixelsArray();
    var p2 = image2.getPixelsArray();
    if (p1.length != p2.length) {
        console.log("Images had two different array lengths.");
        return -1;
    }
    var error_sum = 0;
    for (var i = 0; i < p1.length; i++) {
        for (var j = 0; j < 3; j++) {
            error_sum += Math.pow(p1[i][j] - p2[i][j], 2);
        }
    }
    var mse = error_sum / (p1.length * 3);
    return 20 * Math.log10(255) - 10 * Math.log10(mse);
}

/// Displays a JPEG image that has roughly the target BPM.
function displaySimilarBppJpeg(image, bpp) {
    var view = document.getElementById("jpeg_image");
    var bpp_result = findBpp(bpp, image);
    displayBuffer(view, bpp_result.buffer);
    return bpp_result;
}

function activateDecoderTags() {
    document.getElementById("dec-tag-jpeg").innerHTML = "JPEG:";
    document.getElementById("dec-tag-ours").innerHTML = "Zipnet: ";
}

/// Starts the encoding process and then opens a download prompt for the user.
function downloadEncodedImage() {
    var file = document.getElementById("imgInput").files[0];
    var reader = new FileReader();

    // Fallout if no file is selected beforehand.
    if (file) {
        file.arrayBuffer().then(
            (array_buffer) => {
                var data = new Uint8Array(array_buffer);
                // Measuring exection time for the statistics
                var start = performance.now();
                var encoded = wasm.encode_image(data);
                var end = performance.now();

                Image.load(data).then(
                    (image) => {
                        var bpp = (encoded.length * 8) / getNumPixels(image);
                        displayEncoderStats(data.length, encoded.length, bpp, end - start);
                        last_bpp = bpp;
                    }
                ).catch(imageReadFail);

                promptDownload(encoded);
            }, imageFileError
        )
    }
    else {
        alert("Please first select an image to encode.");
        console.log("Tried to encode without selected image.");
        return;
    }
}

function imageFileError(err) {
    alert("Couldn't read image. Error: " + err);
}

function displayEncoderStats(prev_size, new_size, bpp, time) {
    document.getElementById("statsEncoded").innerHTML = "<p> Previous size: "
        + prev_size + " Byte, Encoded size: " + new_size + " Byte (" + (new_size / prev_size * 100).toFixed(2) + " % of original), " + bpp.toFixed(2) + " bpp" + ", Encoding time: " + time.toFixed() + " ms</p>";
}

function displayDecoderStats(time) {
    document.getElementById("statsDecoded").innerHTML = "<p>Decoding time: " + time.toFixed() + " ms </p>";
}

function displayDecoderStatsExtended(time, bpp_zipnet, bpp_jpeg, psnr_zipnet, psnr_jpeg) {
    document.getElementById("statsDecoded").innerHTML = "<p>Decoding time: " + time.toFixed() + " ms <br><span style='width: 3.2em; display:inline-block'>Zipnet:</span>" + bpp_zipnet.toFixed(2) + " bpp&ensp;" + psnr_zipnet.toFixed(2) + " PSNR <br><span style='width: 3.2em; display:inline-block'>JPEG:</span>" + bpp_jpeg.toFixed(2) + " bpp&ensp;" + psnr_jpeg.toFixed(2) + " PSNR</p>";
}

/// Displays the image at the buffer at the .src attribute of the given preview element.
function displayBuffer(view, buffer) {
    // Puts the decoded data into the preview (in a convoluted way)
    var blob = new Blob([buffer]);
    var view_reader = new FileReader();
    view_reader.onload = function (e) {
        view.height = "200";
        view.src = e.target.result;
        view.style.display = "block";
    }
    view_reader.readAsDataURL(blob);
}

/// Decodes image and displays it in the preview, as well as a JSON representation
/// if possible.
function decodeImageAndPreview() {
    var preview = document.getElementById("decodedPreview");
    var file = document.getElementById('encodedInput').files[0];
    // original is used to calculate PSNR and JPEG reconstructions
    var original_file = document.getElementById("imgInput").files[0];

    if (file) {
        file.arrayBuffer().then(
            (array_buffer) => {
                var data = new Uint8Array(array_buffer);

                // Measuring time for the performance stats.
                var start = performance.now();
                var decoded = wasm.decode_image(data).buffer;
                var end = performance.now();
                displayBuffer(preview, decoded);

                if (original_file && last_bpp) {
                    original_file.arrayBuffer().then(
                        (orig_array_buffer) => Image.load(orig_array_buffer)
                    ).then(
                        (im_orig) => {
                            var jpeg_res = displaySimilarBppJpeg(im_orig, last_bpp);

                            Promise.all([Image.load(jpeg_res.buffer), Image.load(decoded)]).then(
                                (images) => {
                                    var im_jpeg = images[0];
                                    var im_dec = images[1];
                                    var psnr_zipnet = calculatePsnr(im_orig, im_dec);
                                    var psnr_jpeg = calculatePsnr(im_orig, im_jpeg);
                                    activateDecoderTags();
                                    displayDecoderStatsExtended(end - start, last_bpp, jpeg_res.bpp, psnr_zipnet, psnr_jpeg);
                                }
                            )
                        }
                    )
                }
                else {
                    displayDecoderStats(end - start);
                }
            }
        )
    }
}

/// Prompts the user to download the decoded image that is shown in the decoder preview.
function downloadDecodedImage() {
    var img_dataurl = document.getElementById("decodedPreview").src;
    Image.load(img_dataurl).then(
        (_) => {
            var a = document.createElement("a");
            document.body.appendChild(a);
            a.style = "display: none";
            a.href = img_dataurl;
            a.download = "zipnet-decoded-" + Date.now() + ".png";
            a.click();
            window.URL.revokeObjectURL(img_dataurl);
        })
        .catch(
            (error) => {
                alert("Error getting your download ready. Maybe you didn't upload an image or didn't wait for the decoding process to finish?");
                console.log("Download error: " + error);
            });
}

console.log("Reload");
document.getElementById('imgInput').addEventListener("change", previewImageToEncode);
document.getElementById("downloadEncoded").addEventListener("click", downloadEncodedImage);
document.getElementById('encodedInput').addEventListener("change", decodeImageAndPreview);
document.getElementById("downloadDecoded").addEventListener("click", downloadDecodedImage);
