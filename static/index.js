/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const LEMURNET_CLASSES = {
  0: 'Avahi',
  1: 'Cheirogaleus',
  2: 'Daubentonia Madagascariensis',
  3: 'Eulemur Abifrons',
  4: 'Eulemur Cinereicceps',
  5: 'Eulemur Collaris',
  6: 'Eulemur Coronatus',
  7: 'Eulemur Flavifrons',
  8: 'Eulemur Fulvus',
  9: 'Eulemur Macao',
  10: 'Eulemur Mongoz',
  11: 'Eulemur Rubriventer',
  12: 'Eulemur Rufifrons',
  13: 'Eulemur Rufus',
  14: 'Eulemur Sanfordi',
  15: 'Hapalemur Aureus',
  16: 'Hapalemur Griseus',
  17: 'Indri Indri',
  18: 'Lemur Catta',
  19: 'Lepilemur',
  20: 'Microcebus',
  21: 'Mirza',
  22: 'Phaner',
  23: 'Prolemur Simus',
  24: 'Propithecus Candidus',
  25: 'Propithecus Coquereli',
  26: 'Propithecus Coronatus',
  27: 'Propithecus Deckenii',
  28: 'Propithecus Diadema',
  29: 'Propithecus Edwardsi',
  30: 'Propithecus Perrieri',
  31: 'Propithecus Tattersalli',
  32: 'Propithecus Verreauxi',
  33: 'Varecia Rubra',
  34: 'Varecia Variegata'
};

const LEMURNET_COMMON_NAMES_CLASSES = {
  0: 'Woolly Lemur',
  1: 'Dwarf Lemur',
  2: 'Aye-Aye',
  3: 'White-Headed Lemur',
  4: 'Gray-Headed Lemur',
  5: 'Collared Brown Lemur',
  6: 'Crowned Lemur',
  7: 'Blue-Eyed Black Lemur',
  8: 'Common Brown Lemur',
  9: 'Black Lemur',
  10: 'Mongoose Lemur',
  11: 'Red-Bellied Lemur',
  12: 'Red-Fronted Lemur',
  13: 'Red Lemur',
  14: 'Sanford&apos;s Brown Lemur',
  15: 'Golden Bamboo Lemur',
  16: 'Eastern Lesser Bamboo Lemur',
  17: 'Indri',
  18: 'Ring-Tailed Lemur',
  19: 'Sportive Lemur',
  20: 'Mouse Lemur',
  21: 'Giant Mouse Lemur',
  22: 'Fork-Marked Lemur',
  23: 'Greater Bamboo Lemur',
  24: 'Silky Sifaka',
  25: 'Coquerel&apos;s Sifaka',
  26: 'Crowned Sifaka',
  27: 'Von der Decken&apos;s Sifaka',
  28: 'Diademed Sifaka',
  29: 'Milne-Edward&apos;s Sifaka',
  30: 'Perrier&apos;s Sifaka',
  31: 'Golden-Crowned Sifaka',
  32: 'Verreaux&apos;s Sifaka',
  33: 'Red-Ruffed Lemur',
  34: 'Black-And-White Ruffed Lemur'
};

const LEMURNET_CONSERVATION_STATUS = {
  0: 'Endangered',
  1: 'Includes Critically Endangered species',
  2: 'Endangered',
  3: 'Endangered',
  4: 'Critically Endangered',
  5 : 'Endangered',
  6: 'Endangered',
  7: 'Critically Endangered',
  8: 'Near Threatened',
  9: 'Vulnerable',
  10: 'Critically Endangered',
  11: 'Vulnerable',
  12: 'Near Threatened',
  13: 'Vulnerable',
  14: 'Endangered',
  15: 'Critically Endangered',
  16: 'Vulnerable',
  17: 'Critically Endangered',
  18: 'Endangered',
  19: 'Includes Critically Endangered species',
  20: 'Includes Critically Endangered species',
  21: 'Endangered',
  22: 'Includes Endangered species',
  23: 'Critically Endangered',
  24: 'Critically Endangered',
  25: 'Endangered',
  26: 'Endangered',
  27: 'Endangered',
  28: 'Critically Endangered',
  29: 'Endangered',
  30: 'Critically Endangered',
  31: 'Critically Endangered',
  32: 'Endangered',
  33: 'Critically Endangered',
  34: 'Critically Endangered'
};

const MOBILENET_MODEL_PATH = 'static/model.json'
    // tslint:disable-next-line:max-line-length
    //$.post(url_for('static', filename='model.json'));

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 5;

let mobilenet;
const mobilenetDemo = async () => {
  mobilenet = await tf.loadModel(MOBILENET_MODEL_PATH);

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  // Make a prediction through the locally hosted cat.jpg.
  const catElement = document.getElementById('cat');
  if (catElement.complete && catElement.naturalHeight !== 0) {
    predict(catElement);
  } else {
    catElement.onload = () => {
      predict(catElement);
    }
  }
};

/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {
  const startTime = performance.now();
  const logits = tf.tidy(() => {
    // tf.fromPixels() returns a Tensor from an image element.
    const img = tf.fromPixels(imgElement).toFloat();

    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    // Make a prediction through mobilenet.
    return mobilenet.predict(batched);
  });

  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime = performance.now() - startTime;

  // Show the classes in the DOM.
  showResults(imgElement, classes);
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: LEMURNET_CLASSES[topkIndices[i]],
      probability: topkValues[i],
	  commonName: LEMURNET_COMMON_NAMES_CLASSES[topkIndices[i]],
	  IUCN: LEMURNET_CONSERVATION_STATUS[topkIndices[i]]
    })
  }
  return topClassesAndProbs;
}

function showResults(imgElement, classes) {
          
  const predictionContainer = document.createElement('section');
  predictionContainer.className = 'section--center mdl-grid mdl-grid--no-spacing mdl-shadow--2dp';
  
  const predictionHeader = document.createElement('header');
  predictionHeader.className = "section__play-btn mdl-cell mdl-cell--3-col-desktop mdl-cell--2-col-tablet mdl-cell--4-col-phone mdl-color--teal-100 mdl-color-text--white";
  imgElement.style.display = 'block';
  predictionHeader.appendChild(imgElement);
  predictionContainer.appendChild(predictionHeader)

  const resultList = document.createElement("ul");
  resultList.className = "mdl-list";
  resultList.style.paddingTop = 0;
  resultList.style.paddingBottom = 0;
  
  const scientificName = document.createElement("li")
  scientificName.className = "mdl-list__item";
  scientificName.innerHTML = '<span class="mdl-list__item-primary-content"><i class="material-icons mdl-list__item-icon">school</i>' + classes[0].className + '</span>';
  scientificName.style.paddingTop = 0;
  resultList.appendChild(scientificName);
  
  const confidenceLevel = document.createElement("li")
  confidenceLevel.className = "mdl-list__item";
  confidenceLevel.innerHTML = '<span class="mdl-list__item-primary-content"><i class="material-icons mdl-list__item-icon">how_to_reg</i>' + Math.floor(100*classes[0].probability) + '&percnt; confidence</span>';
  confidenceLevel.style.paddingTop = 0;
  resultList.appendChild(confidenceLevel);
  
  const conservationStatus = document.createElement("li")
  conservationStatus.className = "mdl-list__item";
  conservationStatus.innerHTML = '<span class="mdl-list__item-primary-content"><i class="material-icons mdl-list__item-icon">local_hospital</i>' + classes[0].IUCN +'</span>';
  conservationStatus.style.paddingTop = 0;
  resultList.appendChild(conservationStatus);
  
  const predictionCard = document.createElement('div');
  predictionCard.className = "mdl-card mdl-cell mdl-cell--9-col-desktop mdl-cell--6-col-tablet mdl-cell--4-col-phone";

  const predictionTitle = document.createElement('div');
  predictionTitle.className = "mdl-card__title";
  predictionTitle.innerHTML= "<h1 class='mdl-card__title-text'><b>" + classes[0].commonName + "</b></h1>";
  predictionTitle.style.paddingBottom = "4px";
  
  const predictionText = document.createElement('div');
  predictionText.className = "mdl-card__supporting-text";
  predictionText.appendChild(resultList);
  
  predictionCard.appendChild(predictionTitle);
  predictionCard.appendChild(resultList);	
  
  predictionContainer.appendChild(predictionCard);
  
  button_section.style.display =  "block";
  loading.style.display = "none";
  
  overview.insertBefore(
      predictionContainer, overview.childNodes[3]);
}

const filesElement = document.getElementById("input_button");

filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const button_section = document.getElementById("button_section");
const loading = document.getElementById("loading");
const overview = document.getElementById("overview");

const predictionsElement = document.getElementById('predictions');

mobilenetDemo();
