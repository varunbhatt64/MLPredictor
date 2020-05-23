const express = require('express');
const app = express();
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const methodOverride = require('method-override');
const Model = require("./models/model.js");
const modelRoutes = require('./routes/models');
const algorithmRoutes = require('./routes/algorithms');
require('dotenv').config();

app.set("view engine", "ejs");
app.use(express.static("public"));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

app.use(methodOverride("_method")); 

// To suppress warning
mongoose.set('useFindAndModify', false);

// Connect to mongodb
mongoose.connect(process.env.MONGODB_URL_LOCAL, {
    useUnifiedTopology: true,
    useNewUrlParser: true
})
    .then(() => console.log('Db connected'))
    .catch(err => {
        console.log("Db connect Error: $len{err.message}");
    });

//Home Page
app.get("/", function (req, res) {
    res.render("index");
});

app.use(modelRoutes);
app.use("/models", modelRoutes);
app.use(algorithmRoutes);
app.use("/algorithms", algorithmRoutes);

app.listen(process.env.PORT || 3001, function () {
    console.log("ML Predictor Server Started!")
});