const express = require('express');
const Model = require('../models/model');
const Algorithm = require("../models/algorithm.js");

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

//EDIT  Route - GET
router.get("/models/:id/edit", function (req, res) {
    Model.findById(req.params.id).populate('algorithm').exec(function (err, foundModel) {
        if (err)
            res.redirect("/models");
        else {
            Algorithm.find({}, function (err, algorithms) {
                if (err)
                    console.log(err);
                else
                {
                    const data = { algorithms: algorithms, model: foundModel };
                    res.render("editModel", {data : data});
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

module.exports = router;