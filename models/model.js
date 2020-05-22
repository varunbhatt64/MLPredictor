const mongoose = require('mongoose');

let modelSchema = mongoose.Schema({
    name: String,
    algorithm: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Algorithm"
    },
    mlCategory: String,
    inputPath: String,
    isTrained: Boolean,
    isCustom: Boolean,
    description: String
});

let Model = mongoose.model("Model", modelSchema);

module.exports = Model;