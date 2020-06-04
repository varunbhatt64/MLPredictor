const express = require('express');
const Model = require('../models/model');
const Algorithm = require("../models/algorithm.js");
//const tf = require("@tensorflow/tfjs");
const tf = require('@tensorflow/tfjs-node');

let router = express.Router();

///////////////////////
let headers;
let featureNames;
let points;
let numOfFeatures;
let normalizedFeature, normalizedLabel;
///////////////////////

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
        else {
            if (foundModel.algorithm.type === "regression")
                res.render("modelRegression", { model: foundModel });
            else
                res.render("modelClassification", { model: foundModel });
        }
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
router.post("/models/:id", function (req, res) {
    console.log(req.body);
});

router.get("/models/:id/predict", async function (req, res) {
    Model.findById(req.params.id).populate('algorithm').exec(async function (err, foundModel) {
        if (err){
            console.log('model not found');
            res.status(404).json({
                message: "Model not found",
                err
            });
        }
        else {
            const handler = tf.io.fileSystem(`C://Users/varun/Downloads/${req.params.id}.json`);
            const model = await tf.loadLayersModel(handler);
            await run(foundModel.inputPath, foundModel.label);
            let inputs = [];
            console.log(featureNames);
            console.log(req.query);
            featureNames.forEach(element => {
                element = element.trimEnd();
                const predctionInput = parseInt(req.query[element]);
                if (isNaN(predctionInput)) {
                    res.status(400).json({
                        message: "Invalid parameter"                        
                    });
                }
                inputs.push(predctionInput);
            });
            const inputTensor = tf.tensor2d([inputs]);
            const normalizedInput = normalize(inputTensor, normalizedFeature.min, normalizedFeature.max);
            const normalizedOutputTensor = model.predict(normalizedInput.tensor);
            const outputTensor = denormalize(normalizedOutputTensor, normalizedLabel.min, normalizedLabel.max);
            const outputValue = outputTensor.dataSync()[0];
            const outputRoundedValue = outputValue.toFixed(2);
            console.log(`predicted value - ${outputRoundedValue}`);
            try {
                res.status(200).json({
                    data: outputRoundedValue
                });
            } catch (err) {
                res.status(400).json({
                    message: "Some error occured",
                    err
                });
            }
        }
    });
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

async function run(csvUrl, label) {
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
            y: Object.values(ys)
        }
    }).take(1000);
    //console.log(await pointsDataset.toArray());

    points = await pointsDataset.toArray();
    //if number of elements is odd then remove the last element to make split work
    if (points.length % 2 !== 0) {
        points.pop();
    }
    // shuffle the data
    tf.util.shuffle(points);

    // extract Features (inputs)
    const featureValues = points.map(p => p.x);
    const featureTensor = tf.tensor2d(featureValues);

    // extract Labels (outputs)
    const labelValues = points.map(p => p.y);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

    // normalize using min-max
    normalizedFeature = normalize(featureTensor);
    normalizedLabel = normalize(labelTensor);

    featureTensor.dispose();
    labelTensor.dispose();   
}

module.exports = router;