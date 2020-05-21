const mongoose = require('mongoose');

let modelSchema = mongoose.Schema({
    name: String,
    algorithm: String,
    mlCategory: String,
    inputPath: String,
    isTrained: Boolean,
    isCustom: Boolean,
    description: String
});

let Model = mongoose.model("Category", modelSchema);

module.exports = Model;