const express = require('express');
const Algorithm = require("../models/algorithm.js");

let router = express.Router();

// Index - GET
router.get("/", function (req, res) {
    Algorithm.find({}, function (err, algorithms) {
        if (err)
            console.log(err);
        else
            res.render("algorithms", { algorithms: algorithms });
    });
});

//Add algorithm - Post
router.post("/", function (req, res) {
    // add algorithm
    Algorithm.create(req.body.algorithm, function (err, newAlgorithm) {
        if (err) {
            console.log(err);
            res.render("newAlgorithm");
        }
        else {
            res.redirect("/algorithms");
        }
    });
});

//New algorithm- GET
router.get("/new", function (req, res) {
    res.render("newAlgorithm");
});

//EDIT  Route - GET
router.get("/:id/edit", function (req, res) {
    Algorithm.findById(req.params.id, function (err, foundAlgorithm) {
        if (err)
            res.redirect("/algorithms");
        else
            res.render("editAlgorithm", { algorithm: foundAlgorithm });
    });
});


//UPDATE Route-PUT
router.put("/:id", function (req, res) {
    Algorithm.findByIdAndUpdate(req.params.id, req.body.algorithm, function (err, updatedAlgorithm) {
        if (err)
            res.redirect("/algorithms" + req.params.id + "/edit");
        else
            res.redirect("/algorithms");
    });
});

//DELETE Route - DELETE
router.delete("/:id", function (req, res) {
    Algorithm.findByIdAndRemove(req.params.id, function (err) {
        if (err)
            res.redirect("/algorithms");
        else
            res.redirect("/algorithms");
    });
});

module.exports = router;