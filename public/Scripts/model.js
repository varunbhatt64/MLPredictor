let model;
let normalizedFeature, normalizedLabel;
let trainingFeatureTensor, testingFeatureTensor, trainingLabelTensor, testingLabelTensor;

run();

async function save() {

}

async function test() {
    const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
    const loss = await lossTensor.dataSync();
    console.log(`Testing set loss: ${loss}`);
    $('#testing-status').html(
        `Testing set loss: ${loss.toPrecision(5)}`
    );
}

async function train() {
    //disable all the buttons and change training status
    $('#main').find(':button').prop('disabled', true);
    $('#model-status').html('Training...');

    model = createModel();
    tfvis.show.modelSummary({ name: `Model Summary`/*, tab: `Model`*/ }, model);
    const layer = model.getLayer(undefined, 0);
    //tfvis.show.layer({ name: `Layer 1`, tab: `Model Inspection` }, layer);

    const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor);
    const trainingLoss = result.history.loss.pop();
    console.log(`Training set loss: ${trainingLoss}`);

    const validationLoss = result.history.val_loss.pop();
    console.log(`Validation set loss: ${validationLoss}`);
    $('#model-status').html(`Trained {unsaved}\n`
        + `Training set loss: ${trainingLoss.toPrecision(5)}\n`
        + `Validation set loss: ${validationLoss.toPrecision(5)}`);

    $('#test-button').removeAttr('disabled');
}

async function toggleVisor() {
    tfvis.visor().toggle();
}

function normalize(tensor) {
    const min = tensor.min();
    const max = tensor.max();
    const normalizedTensor = tensor.sub(min).div(max.sub(min));
    return {
        tensor: normalizedTensor,
        min: min,
        max: max
    };
}

function denormalize(tensor, min, max) {
    const denormalizedTensor = tensor.mul(max.sub(min)).add(min);
    return denormalizedTensor;
}

function createModel() {
    model = tf.sequential();
    model.add(tf.layers.dense({
        units: 1,
        useBias: true,
        activation: 'linear',
        inputDim: 1,
    }));
    // define optimizer
    const optimizer = tf.train.sgd(0.1);

    //compile the model
    model.compile({
        loss: 'meanSquaredError',
        optimizer,
    });

    return model;
}

async function trainModel(model, trainingFeatureTensor, trainingLabelTensor) {
    const { onBatchEnd, onEpochEnd } = tfvis.show.fitCallbacks(
        { name: "Training Performance" },
        ['loss']
    );

    return model.fit(trainingFeatureTensor, trainingLabelTensor, {
        batchSize: 32,
        epochs: 20,
        validationSplit: 0.2,
        // callbacks: {
        //     onEpochEnd: async (epoch, log) => {
        //         console.log(`epoch ${epoch}: loss = ${log.loss}`);
        //     }
        callbacks: {
            onBatchEnd,
            onEpochEnd,
        }
    });
}

function plot(points, featureName, labelName) {
    tfvis.render.scatterplot(
        { name: `${featureName} vs ${labelName}` },
        { values: [points], series: ['original'] },
        {
            xLabel: featureName,
            yLabel: labelName
        }
    );
}

async function run() {
    //import csv file
    const csvUrl = "http://192.168.1.226:8080/kc_house_data.csv";
    const csvDataset = tf.data.csv(
        csvUrl, {
        columnConfigs: {
            price: {
                isLabel: true
            }
        }
    });
    //console.log(await csvDataset.take(1).toArray());

    //extract feature and label
    const pointsDataset = csvDataset.map(({ xs, ys }) => {
        return {
            x: xs.sqft_living,
            y: ys.price
        }
    });
    //console.log(await pointsDataset.toArray());

    const points = await pointsDataset.toArray();
    //if number of elements is odd then remove the last element to make split work
    if (points.length % 2 !== 0) {
        points.pop();
    }
    // shuffle the data
    tf.util.shuffle(points);

    plot(points, "Square Feet", "Price");

    // extract Features (inputs)
    const featureValues = points.map(p => p.x);
    const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

    // extract Labels (outputs)
    const labelValues = points.map(p => p.x);
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

    // Update status and enable train button
    $('#model-status').html("Model not trained");
    $('#train-button').removeAttr('disabled');
}

async function predict() {

}

async function load() {

}