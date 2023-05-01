let audioCtx = null;
let inputBuffer = null;
let outputBuffer = null;
let isRecording = false;
let stream = null;
let recorder = null;
let chunks = [];

// ENABLING AUDIO CONTEXT
const enableAudioCtx = () => {
  if (audioCtx != null) return;
  console.log("enabling audio");
  audioCtx = new (AudioContext || webkitAudioContext)();
};

// UPLOAD VARIOUS SOURCES TO MEMORY
const toogleRecording = async () => {
  console.log("toogle recording");
  let recordButton = document.getElementById("record-button");
  if (!isRecording) {
    recordButton.value = "Stop recording";
    isRecording = true;
    chunks = [];
    try {
      // GET INPUT STREAM AND CREATE RECORDER
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recorder = new MediaRecorder(stream);

      // ON STOP FUNCTION
      recorder.onstop = function (e) {
        let blob = new Blob(chunks, { type: "audio/ogg; codecs=opus" });
        chunks = [];
        let audioURL = URL.createObjectURL(blob);

        urlToBuffer(audioURL).then((buffer) => {
          inputBuffer = buffer;
          let playButton = document.getElementById("play_input");
          let ravifyButton = document.getElementById("ravify_button");
          playButton.disabled = false;
          ravifyButton.disabled = false;
        });
      };

      recorder.ondataavailable = function (e) {
        chunks.push(e.data);
      };

      recorder.start();
    } catch (err) {
      console.log(err);
    }
  } else {
    recordButton.value = "Record from microphone";
    isRecording = false;
    recorder.stop();
    stream.getTracks().forEach((track) => track.stop());
  }
};

const loadUploadedFile = () => {
  enableAudioCtx();

  let fileInput = document.getElementById("audio-file");
  let ravifyButton = document.getElementById("ravify_button");
  if (fileInput.files[0] == null) return;

  let playButton = document.getElementById("play_input");
  playButton.disabled = true;
  var reader1 = new FileReader();
  reader1.onload = function (ev) {
    audioCtx.decodeAudioData(ev.target.result).then(function (buffer) {
      buffer = tensorToBuffer(bufferToTensor(buffer));
      inputBuffer = buffer;
      playButton.disabled = false;
      ravifyButton.disabled = false;
    });
  };
  reader1.readAsArrayBuffer(fileInput.files[0]);
};

const loadCantina = async () => {
  let playButton = document.getElementById("play_input");
  let ravifyButton = document.getElementById("ravify_button");
  playButton.disabled = true;

  let buffer = await urlToBuffer("/ravejs/default.mp3");
  buffer = tensorToBuffer(bufferToTensor(buffer));
  inputBuffer = buffer;
  playButton.disabled = false;
  ravifyButton.disabled = false;
};

const urlToBuffer = async (url) => {
  enableAudioCtx();
  const audioBuffer = await fetch(url)
    .then((res) => res.arrayBuffer())
    .then((ArrayBuffer) => audioCtx.decodeAudioData(ArrayBuffer));
  return audioBuffer;
};

// PLAY BUFFER
const playBuffer = (buffer) => {
  enableAudioCtx();
  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);
  source.start();
};

const playInput = () => {
  if (inputBuffer == null) return;
  playBuffer(inputBuffer);
};

const playOutput = () => {
if (outputBuffer == null) return;
playBuffer(outputBuffer);
};

// PROCESSING
const transfer = async () => {
if (inputBuffer == null) return;
console.log("transfer in progress...");
outputBuffer = await raveForward(inputBuffer);
make_download(outputBuffer, outputBuffer.getChannelData(0).length);
};

const bufferToTensor = (buffer) => {
let b = buffer.getChannelData(0);
let full_length = b.length;
const inputTensor = new ort.Tensor("float32", b, [1, 1, full_length]);
return inputTensor;
};

const tensorToBuffer = (tensor) => {
let len = tensor.dims[2];
let buffer = audioCtx.createBuffer(1, len, audioCtx.sampleRate);
channel = buffer.getChannelData(0);
for (let i = 0; i < buffer.length; i++) {
channel[i] = isNaN(tensor.data[i]) ? 0 : tensor.data[i];
}
return buffer;
};
/*
const raveForward = async (buffer) => {
let model_name = document.getElementById("model");
let playButton = document.getElementById("play_output");
let ravifyButton = document.getElementById("ravify_button");
ravifyButton.disabled = true;
playButton.disabled = true;
let inputTensor = bufferToTensor(buffer);
let session = await ort.InferenceSession.create(model_name.value);
let feeds = { audio_in: inputTensor };
let audio_out = (await session.run(feeds)).audio_out;
audio_out = tensorToBuffer(audio_out);
playButton.disabled = false;
ravifyButton.disabled = false;
return audio_out;
};
*/

// NEW FUNCTION: tensorToStereoBuffer
const tensorToStereoBuffer = (leftTensor, rightTensor) => {
  let len = leftTensor.dims[2];
  let buffer = audioCtx.createBuffer(2, len, audioCtx.sampleRate);
  let leftChannel = buffer.getChannelData(0);
  let rightChannel = buffer.getChannelData(1);

  for (let i = 0; i < buffer.length; i++) {
    leftChannel[i] = isNaN(leftTensor.data[i]) ? 0 : leftTensor.data[i];
    rightChannel[i] = isNaN(rightTensor.data[i]) ? 0 : rightTensor.data[i];
  }

  return buffer;
};

const raveForward = async (buffer) => {
  let model_name = document.getElementById("model");
  let playButton = document.getElementById("play_output");
  let ravifyButton = document.getElementById("ravify_button");
  ravifyButton.disabled = true;
  playButton.disabled = true;
  let inputTensor = bufferToTensor(buffer);
  let noisyInputTensor = noiseToTensor(buffer);

  let session = await ort.InferenceSession.create(model_name.value);
  let feeds = { audio_in: inputTensor };
  let audio_out = (await session.run(feeds)).audio_out;

  feeds = { audio_in: noisyInputTensor };
  let noisy_audio_out = (await session.run(feeds)).audio_out;

  audio_out = tensorToStereoBuffer(audio_out, noisy_audio_out);

  playButton.disabled = false;
  ravifyButton.disabled = false;
  return audio_out;
};

// NEW FUNCTION: noiseToTensor
const noiseToTensor = (buffer, noiseLevel = 0.005) => {
  const source = buffer.getChannelData(0);
  const noisyData = new Float32Array(source.length);

  for (let i = 0; i < source.length; i++) {
    noisyData[i] = source[i] + (Math.random() * 2 - 1) * noiseLevel;
  }

  const noisyTensor = new ort.Tensor("float32", noisyData, [1, 1, noisyData.length]);
  return noisyTensor;
};

const applyFilters = async (buffer, lowpassFreq, highpassFreq) => {
  const lowpassFilter = audioCtx.createBiquadFilter();
  lowpassFilter.type = "lowpass";
  lowpassFilter.frequency.setValueAtTime(lowpassFreq, audioCtx.currentTime);

  const highpassFilter = audioCtx.createBiquadFilter();
  highpassFilter.type = "highpass";
  highpassFilter.frequency.setValueAtTime(highpassFreq, audioCtx.currentTime);

  const source = audioCtx.createBufferSource();
  source.buffer = buffer;

  source.connect(highpassFilter);
  highpassFilter.connect(lowpassFilter);

  const offlineCtx = new OfflineAudioContext(buffer.numberOfChannels, buffer.length, buffer.sampleRate);
  lowpassFilter.connect(offlineCtx.destination);

  return new Promise((resolve) => {
    source.start(0);
    offlineCtx.startRendering().then((renderedBuffer) => {
      resolve(renderedBuffer);
    });
  });
};


const processAudio = async () => {
  if (inputBuffer === null) return;

  const lowpassFreq = parseFloat(document.getElementById("lowpass").value);
  const highpassFreq = parseFloat(document.getElementById("highpass").value);
  
  const monoBuffer = audioCtx.createBuffer(1, inputBuffer.length, inputBuffer.sampleRate);
  monoBuffer.copyToChannel(inputBuffer.getChannelData(0), 0);
  
  const filteredBuffer = await applyFilters(monoBuffer, lowpassFreq, highpassFreq);
  outputBuffer = await raveForward(filteredBuffer);
  make_download(outputBuffer, outputBuffer.getChannelData(0).length);
};
