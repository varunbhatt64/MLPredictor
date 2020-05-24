// load the MobileNet
let imageClassifier;

(async function(){
  imageClassifier = ml5.imageClassifier('MobileNet', modelReady);
})();

// when model is ready make predictions
function modelReady() {
  document.getElementById('status').innerHTML = 'Model Loaded';
}

$('#image-selector').change(function(){
  let reader = new FileReader();
  reader.onload = function(){
    let dataURL = reader.result;
    $('#img').attr("src", dataURL);
    $('#prediction-list').empty();
  }
  let file = $('#image-selector').prop('files')[0];
  reader.readAsDataURL(file);
});

$('#predict').click(async function(){
  let prediction = imageClassifier.predict(img, gotResults);
})

function gotResults(error, results) {
  $('#prediction-list').empty();
  if (error) {
      console.error(error);
  } else {
      results.forEach(element => {
        $('#prediction-list').append(`<li>${element.label}: ${element.confidence.toFixed(2) * 100} %</li>`);
      });
      console.log(results);    
  }
}