const express = require('express');
const Model = require('../models/model');
const Algorithm = require("../models/algorithm.js");
const tf = require('@tensorflow/tfjs-node');

let router = express.Router();

// Index - GET
router.get("/models", function (req, res) {
    Model.find({}).populate('algorithm').exec(function (err, models) {
        if (err)
            console.log(err);
        else
            res.render("models", { models: models });
    });
});

//New model- GET
router.get("/models/new", function (req, res) {
    Algorithm.find({}, function (err, algorithms) {
        if (err)
            console.log(err);
        else
            res.render("newModel", { algorithms: algorithms });
    });
});

//Custom model- GET
router.get("/models/custom", function (req, res) {
    Model.find({ isCustom: true }, function (err, models) {
        if (err)
            console.log(err);
        else
            res.render("models", { models: models });
    });
});

//Trained model- GET
router.get("/models/trained", function (req, res) {
    Model.find({ isTrained: true }, function (err, models) {
        if (err)
            console.log(err);
        else
            res.render("models", { models: models });
    });
});

//Model- GET
router.get("/models/:id", function (req, res) {
    Model.findById(req.params.id).populate('algorithm').exec(function (err, foundModel) {
        if (err)
            res.redirect("/models");
        else
            res.render("model", { model: foundModel });
    });
});

//Create model - Post
router.post("/models", function (req, res) {
    // create model
    Model.create(req.body.model, function (err, newModel) {
        if (err) {
            console.log(err);
            res.render("newModel");
        }
        else {
            //redirect to the index
            res.redirect("/models");
        }
    });
});

//Load data - Post
router.post("/models/:id", async function (req, res) {
    // const csvDataset = tf.data.csv(
    //     req.body.url, {
    //     columnConfigs: {
    //         price: {
    //             isLabel: true
    //         }
    //     }
    // });
    //let columnNames = await csvDataset.take(10).toArray();
    //console.log(columnNames);
    await run(req.body.url, 'price');

    // const points = csvDataset.map(record => ({
    //     x: record.sqft_living15,
    //     y: record.price
    // }));
    //const pointsArr = await points.toArray();
    //console.log(pointsArr);
    //tfvs.render.table
    //res.render('vis');
});

//EDIT  Route - GET
router.get("/models/:id/edit", function (req, res) {
    Model.findById(req.params.id).populate('algorithm').exec(function (err, foundModel) {
        if (err)
            res.redirect("/models");
        else {
            Algorithm.find({}, function (err, algorithms) {
                if (err)
                    console.log(err);
                else {
                    const data = { algorithms: algorithms, model: foundModel };
                    res.render("editModel", { data: data });
                }
            });
        }
    });
});

//UPDATE Route-PUT
router.put("/models/:id", function (req, res) {
    Model.findByIdAndUpdate(req.params.id, req.body.model, function (err, updateModel) {
        if (err)
            res.redirect("/models" + req.params.id + "/edit");
        else
            res.redirect("/models");
    });
});

//DELETE Route - DELETE
router.delete("/models/:id", function (req, res) {
    Model.findByIdAndRemove(req.params.id, function (err) {
        if (err)
            res.redirect("/models");
        else
            res.redirect("/models");
    });
});

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
    const model = tf.sequential();
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
    return model.fit(trainingFeatureTensor, trainingLabelTensor, {
        batchSize: 32,
        epochs: 20,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: async (epoch, log) => {
                console.log(`epoch ${epoch}: loss = ${log.loss}`);
            }
        }
    });
}

async function run(csvUrl, label) {
    //import csv file
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

    // extract Features (inputs)
    const featureValues = points.map(p => p.x);
    const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

    // extract Labels (outputs)
    const labelValues = points.map(p => p.x);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

    //featureTensor.print();
    //labelTensor.print();

    // normalize using min-max
    const normalizedFeature = normalize(featureTensor);
    const normalizedLabel = normalize(labelTensor);

    //normalizedFeature.tensor.print();
    //normalizedLabel.tensor.print();

    //denormalize(normalizedFeature.tensor, normalizedFeature.min, normalizedFeature.max).print();

    // split dataset into testing and training - split doesn't work if number of elements are odd
    const [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalizedFeature.tensor, 2);
    const [trainingLabelTensor, testingLabelTensor] = tf.split(normalizedLabel.tensor, 2);

    //trainingFeatureTensor.print();

    const model = createModel();
    //const layer = model.getLayer(undefined, 0);
    //model.summary();
    const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor);
    const trainingLoss = result.history.loss.pop();
    console.log(`Training set loss: ${trainingLoss}`);

    const validationLoss = result.history.val_loss.pop();
    console.log(`Validation set loss: ${validationLoss}`);

    const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
    const loss = await lossTensor.dataSync();
    console.log(`Testing set loss: ${loss}`);
}

module.exports = router;