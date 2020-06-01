let model;
let headers;
let featureNames;
let points;
let numOfFeatures;
let normalizedFeature, normalizedLabel;
let trainingFeatureTensor, testingFeatureTensor, trainingLabelTensor, testingLabelTensor;

const csvUrl = $('#input').val();
const algorithm = $('#algorithm').val();
const label = $('#label').val();
const storageId = $('#modelId').val();
const storageKey = `localstorage://${storageId}`;

run();

async function save() {
    const savedResults = await model.save(storageKey);

    $('#model-status').html(`Trained (saved ${savedResults.modelArtifactsInfo.dateSaved})`);
    $('#save-button').prop('disabled', true);
}

async function load() {
    const models = await tf.io.listModels();
    const modelInfo = models[storageKey];
    if (modelInfo) {
        model = await tf.loadLayersModel(storageKey);
        tfvis.show.modelSummary({ name: `Model Summary`/*, tab: `Model`*/ }, model);

        //await plotPredictionHeatmap();

        $('#model-status').html(`Trained (saved ${modelInfo.dateSaved})`);
        $('#load-button').prop('disabled', true);
        $('#test-button').removeAttr('disabled');
        $('#predict-button').removeAttr('disabled');
    }
    else {
        alert("Could not load: no saved model found");
    }

}

async function predict() {
    //const predctionInput = parseInt($('#prediction-input').val());
    let inputs = [];
    headers.forEach(element => {
        if(element !== label)
        {
            element = element.trimEnd();
            const predctionInput = parseInt($(`#${element}`).val());
            if (isNaN(predctionInput)) {
                alert(`Please enter a valid number for ${element}`);
                return;
            }
            else
                inputs.push(predctionInput);
        }
    });
    console.log(`Inputs - ${inputs}`);
    if (inputs.length === numOfFeatures) {
        tf.tidy(() => {
            const inputTensor = tf.tensor2d([inputs]);
            const normalizedInput = normalize(inputTensor, normalizedFeature.min, normalizedFeature.max);
            const normalizedOutputTensor = model.predict(normalizedInput.tensor);
            const outputTensor = denormalize(normalizedOutputTensor, normalizedLabel.min, normalizedLabel.max);
            const outputValue = outputTensor.dataSync()[0];
            const outputValuePercent = (outputValue*100).toFixed(1);
            console.log(outputValuePercent);
            $('#prediction-output').html(`The likelihood of being a ${label} is ${outputValuePercent}%`);
        });
    }
}

async function test() {
    const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
    const loss = await lossTensor.dataSync();
    console.log(`Testing set loss: ${loss}`);
    $('#testing-status').html(
        `Testing set loss: ${loss/*.toPrecision(5)*/}`
    );
}

async function train() {
    //disable all the buttons and change training status
    $('#main').find(':button').prop('disabled', true);
    $('#model-status').html('Training...');

    model = createModel();
    tfvis.show.modelSummary({ name: `Model Summary`/*, tab: `Model`*/ }, model);
    const layer = model.getLayer(undefined, 0);
    //tfvis.show.layer({ name: "Layer 1"/*, tab: `Model Inspection`*/ }, layer);

    const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor);
    const trainingLoss = result.history.loss.pop();
    console.log(`Training set loss: ${trainingLoss}`);

    const validationLoss = result.history.val_loss.pop();
    console.log(`Validation set loss: ${validationLoss}`);

    //await plotPredictionHeatmap();

    $('#model-status').html(`Trained {unsaved}\n`
        + `Training set loss: ${trainingLoss.toPrecision(5)}\n`
        + `Validation set loss: ${validationLoss.toPrecision(5)}`);

    $('#train-button').removeAttr('disabled');
    $('#test-button').removeAttr('disabled');
    $('#load-button').removeAttr('disabled');
    $('#save-button').removeAttr('disabled');
    $('#predict-button').removeAttr('disabled');
}

async function toggleVisor() {
    tfvis.visor().toggle();
}

function normalize(tensor, previousMin = null, previousMax = null) {
    const tensorDimention = tensor.shape.length > 1 && tensor.shape[1];

    if (tensorDimention && tensorDimention > 1) {
        // more than 1 feature
        // split into separate tensors
        const features = tf.split(tensor, tensorDimention, 1);

        // normalize and find min/max for each feature
        const normalizedFeatures = features.map((featureTensor, i) =>
            normalize(featureTensor,
                previousMin ? previousMin[i] : null,
                previousMax ? previousMax[i] : null
            )
        );

        // now again concat the separate feature tensors to return
        const returnTensor = tf.concat(normalizedFeatures.map(f => f.tensor), 1);
        const min = normalizedFeatures.map(f => f.min);
        const max = normalizedFeatures.map(f => f.max);

        return { tensor: returnTensor, min, max };
    }
    else {
        const min = previousMin || tensor.min();
        const max = previousMax || tensor.max();
        const normalizedTensor = tensor.sub(min).div(max.sub(min));
        return {
            tensor: normalizedTensor,
            min: min,
            max: max
        };
    }
}

function denormalize(tensor, min, max) {
    const tensorDimention = tensor.shape.length > 1 && tensor.shape[1];

    if (tensorDimention && tensorDimention > 1) {
        // more than 1 feature
        // split into separate tensors
        const features = tf.split(tensor, tensorDimention, 1);

        const denormalized = features.map((featureTensor, i) =>
            denormalize(featureTensor, min[i], max[i])
        );

        const returnTensor = tf.concat(denormalized, 1);
        return returnTensor;
    }

    else {
        const denormalizedTensor = tensor.mul(max.sub(min)).add(min);
        return denormalizedTensor;
    }
}

function createModel() {
    const learningRate = $('#learning-rate').val();
    const optimizerFunc = $('#optimizer').val();
    const lossMethod = $('#loss-method').val();
    const numLayers = parseInt($('#layers').val());

    console.log(`learning rate - ${learningRate} optimizer - ${optimizerFunc} loss method - ${lossMethod} layers - ${numLayers}`);

    model = tf.sequential();
    
    const activation = algorithm.includes('Linear') ? 'linear' : 'sigmoid';

    for (let index = 0; index < numLayers - 1; index++) {
        model.add(tf.layers.dense({
            units: 10,
            useBias: true,
            activation: activation,
            inputDim: index === 0 ? numOfFeatures : 10,
        }));
    }

    // output layer
    model.add(tf.layers.dense({
        units: 1,
        useBias: true,
        activation: activation,
        inputDim: numLayers === 1 ? numOfFeatures : 10,
    }));

    // define optimizer
    const optimizer = tf.train.adam(0.1);

    //compile the model
    model.compile({
        loss: 'binaryCrossentropy',
        optimizer,
        //metrics: ['accuracy']
    });

    return model;
}

async function trainModel(model, trainingFeatureTensor, trainingLabelTensor) {
    const { onBatchEnd, onEpochEnd } = tfvis.show.fitCallbacks(
        { name: "Training Performance" },
        ['loss']
    );

    const batchSize = parseInt($('#batch-size').val());
    const epochs = parseInt($('#epochs').val());
    const validationSetSize = parseInt($('#validation-size').val())/100;

    return model.fit(trainingFeatureTensor, trainingLabelTensor, {
        batchSize: batchSize,
        epochs: epochs,
        validationSplit: validationSetSize,
        // callbacks: {
        //     onEpochEnd: async (epoch, log) => {
        //         console.log(`epoch ${epoch}: loss = ${log.loss}`);
        //     }
        callbacks: {
            onBatchEnd,
            onEpochEnd,
            onEpochBegin: async function () {
                //await plotPredictionHeatmap();
            }
        }
    });
}

async function plotClasses (pointsArray, classKey, size = 400, equalizeClassSizes = false) {
    // Add each class as a series
    const allSeries = {};
    pointsArray.forEach(p => {
      // Add each point to the series for the class it is in
      const seriesName = `${classKey}: ${p.class}`;
      let series = allSeries[seriesName];
      if (!series) {
        series = [];
        allSeries[seriesName] = series;
      }
      series.push(p);
    });
  
    if (equalizeClassSizes) {
      // Find smallest class
      let maxLength = null;
      Object.values(allSeries).forEach(series => {
        if (maxLength === null || series.length < maxLength && series.length >= 100) {
          maxLength = series.length;
        }
      });
      // Limit each class to number of elements of smallest class
      Object.keys(allSeries).forEach(keyName => {
        allSeries[keyName] = allSeries[keyName].slice(0, maxLength);
        if (allSeries[keyName].length < 100) {
          delete allSeries[keyName];
        }
      });
    }
  
    tfvis.render.scatterplot(
      {
        name: `Square feet vs House Price`,
        styles: { width: "100%" }
      },
      {
        values: Object.values(allSeries),
        series: Object.keys(allSeries),
      },
      {
        xLabel: "Square feet",
        yLabel: "Price",
        height: size,
        width: size*1.5,
      }
    );
  }

  async function plotPredictionHeatmap (name = "Predicted class", size = 400) {
    const [ valuesPromise, xTicksPromise, yTicksPromise ] = tf.tidy(() => {
      const gridSize = 50;
      const predictionColumns = [];
      // Heatmap order is confusing: columns first (top to bottom) then rows (left to right)
      // We want to convert that to a standard cartesian plot so invert the y values
      for (let colIndex = 0; colIndex < gridSize; colIndex++) {
        // Loop for each column, starting from the left
        const colInputs = [];
        const x = colIndex / gridSize;
        for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
          // Loop for each row, starting from the top
          const y = (gridSize - rowIndex) / gridSize;
          colInputs.push([x, y]);
        }
        
        const colPredictions = model.predict(tf.tensor2d(colInputs));
        predictionColumns.push(colPredictions);
      }
      const valuesTensor = tf.stack(predictionColumns);

      const normalizedLabelsTensor = tf.linspace(0, 1, gridSize);
      const xTicksTensor = denormalize(normalizedLabelsTensor,
        normalizedFeature.min[0], normalizedFeature.max[0]);
      const yTicksTensor = denormalize(normalizedLabelsTensor.reverse(),
        normalizedFeature.min[1], normalizedFeature.max[1]);

      return [ valuesTensor.array(), xTicksTensor.array(), yTicksTensor.array() ];
    });

    const values = await valuesPromise;
    const xTicks = await xTicksPromise;
    const xTickLabels = xTicks.map(l => (l/1000).toFixed(1)+"k sqft");
    const yTicks = await yTicksPromise;
    const yTickLabels = yTicks.map(l => "$"+(l/1000).toFixed(0)+"k");
    const data = {
      values,
      xTickLabels,
      yTickLabels,
    };

    tfvis.render.heatmap({
      name: `${name} (local)`,
      tab: "Predictions"
    }, data, { height: size });
    tfvis.render.heatmap({ 
      name: `${name} (full domain)`, 
      tab: "Predictions" 
    }, data, { height: size, domain: [0, 1] });
}

async function run() {
    //import csv file
    const csvDataset = tf.data.csv(
        csvUrl, {
        columnConfigs: {
            [label]: {
                isLabel: true
            }
        }
    });
    //console.log(await csvDataset.take(1).toArray());

    headers = await csvDataset.columnNames();
    featureNames = headers.filter(value => value !== label);
    console.log(`headers - ${headers} features - ${featureNames}`);
    numOfFeatures = featureNames.length;

    //extract feature and label
    const pointsDataset = csvDataset.map(({ xs, ys }) => {
        return {
            x: Object.values(xs),
            class: Object.values(ys)
        }
    });
    //console.log(await pointsDataset.toArray());

    points = await pointsDataset.toArray();
    //if number of elements is odd then remove the last element to make split work
    if (points.length % 2 !== 0) {
        points.pop();
    }
    // shuffle the data
    tf.util.shuffle(points);

    //plot(points, "Square Feet", label);

    // extract Features (inputs)
    const featureValues = points.map(p => p.x);
    const featureTensor = tf.tensor2d(featureValues);

    // extract Labels (outputs)
    const labelValues = points.map(p => p.class);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

    //featureTensor.print();
    //labelTensor.print();

    // normalize using min-max
    normalizedFeature = normalize(featureTensor);
    normalizedLabel = normalize(labelTensor);

    featureTensor.dispose();
    labelTensor.dispose();

    //normalizedFeature.tensor.print();
    //normalizedLabel.tensor.print();

    //denormalize(normalizedFeature.tensor, normalizedFeature.min, normalizedFeature.max).print();

    // split dataset into testing and training - split doesn't work if number of elements are odd
    [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalizedFeature.tensor, 2);
    [trainingLabelTensor, testingLabelTensor] = tf.split(normalizedLabel.tensor, 2);

    // trainingLabelTensor.print();
    // trainingFeatureTensor.print();

    // Add controls for features
    headers.forEach(element => {
        if(element !== label){
            element = element.trimEnd();
            $('#features').append(`<label>${element.toUpperCase()}: <input type="number" id="${element}"/></label>`);
        }
      });

    // Update status and enable train button
    $('#model-status').html("Model not trained");
    $('#train-button').removeAttr('disabled');
    $('#load-button').removeAttr('disabled');
}