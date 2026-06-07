const fileInput = document.getElementById("file");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let image = null;
let session = null;

async function loadModel() {
    session = await ort.InferenceSession.create(
        "models/realesrgan-x4.onnx"
    );
    console.log("Модель загружена");
}

loadModel();

fileInput.addEventListener("change", e => {

    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = ev => {

        image = new Image();

        image.onload = () => {
            canvas.width = image.width;
            canvas.height = image.height;
            ctx.drawImage(image,0,0);
        };

        image.src = ev.target.result;
    };

    reader.readAsDataURL(file);
});

document.getElementById("upscaleBtn").onclick = async () => {

    if (!image || !session) return;

    const temp = document.createElement("canvas");
    temp.width = image.width;
    temp.height = image.height;

    const tctx = temp.getContext("2d");
    tctx.drawImage(image,0,0);

    const imgData = tctx.getImageData(
        0,
        0,
        temp.width,
        temp.height
    );

    const input = new Float32Array(
        temp.width * temp.height * 3
    );

    let p = 0;

    for(let i=0;i<imgData.data.length;i+=4){

        input[p++] = imgData.data[i] / 255;
        input[p++] = imgData.data[i+1] / 255;
        input[p++] = imgData.data[i+2] / 255;
    }

    const tensor = new ort.Tensor(
        "float32",
        input,
        [1,3,temp.height,temp.width]
    );

    const result = await session.run({
        input: tensor
    });

    console.log(result);

    alert("Апскейл завершён");
};
