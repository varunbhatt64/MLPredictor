let imageClassifier;
 // load the MobileNet
imageClassifier = ml5.imageClassifier('MobileNet', modelReady);
let input = document.querySelector('input[type=file]'); // see Example 4

input.onchange = function () {
  let file = input.files[0];
  let imgURL = URL.createObjectURL(file)
  document.getElementById('img').src = imgURL;
};

let button = document.getElementById('predict');
button.addEventListener("click", predict);

function predict(){
  let prediction = imageClassifier.predict(img, gotResults);
}

function gotResults(error, results) {
  if (error) {
      console.error(error);
  } else {
      console.log(results);    
  }
}

// when model is ready make predictions
function modelReady() {
  document.getElementById('status').innerHTML = 'Model Loaded';
}