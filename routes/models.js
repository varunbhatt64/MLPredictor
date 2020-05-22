const express = require('express');
const Model = require('../models/model');
const Algorithm = require("../models/algorithm.js");

let router = express.Router();

// Index - GET
router.get("/", function (req, res) {
    Model.find({}, function (err, models) {
        if (err)
            console.log(err)
        else
            res.render("models", { models: models })
    });
});


//New model- GET
router.get("/new", function (req, res) {
    Algorithm.find({}, function (err, algorithms) {
        if (err)
            console.log(err)
        else
            res.render("newModel", { algorithms: algorithms })
    });
});

//Custom model- GET
router.get("/custom", function (req, res) {
    Model.find({isCustom: true}, function (err, models) {
        if (err)
            console.log(err)
        else
            res.render("models", { models: models })
    });
});

//Custom model- GET
router.get("/trained", function (req, res) {
    Model.find({isTrained: true}, function (err, models) {
        if (err)
            console.log(err)
        else
            res.render("models", { models: models })
    });
});

//Custom model- GET
router.get("/id", function (req, res) {
    res.render("models", {id: req.params.id});
});

//Create model - Post
router.post("/", function (req, res) {
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
router.get("/:id/edit", function (req, res) {
    Model.findById(req.params.id, function (err, foundModel) {
        if (err)
            res.redirect("/models");
        else
            res.render("editModel", { model: foundModel });
    });
});


//UPDATE Route-PUT
router.put("/:id", function (req, res) {
    Model.findByIdAndUpdate(req.params.id, req.body.model, function (err, updateModel) {
        if (err)
            res.redirect("/models" + req.params.id + "/edit");
        else
            res.redirect("/models");
    });
});

//DELETE Route - DELETE
router.delete("/:id", function (req, res) {
    Model.findByIdAndRemove(req.params.id, function (err) {
        if (err)
            res.redirect("/models");
        else
            res.redirect("/models");
    });
});

module.exports = router;