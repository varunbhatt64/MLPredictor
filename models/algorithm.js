const mongoose = require('mongoose');

let algorithmSchema = mongoose.Schema({
    name: String,    
    type: String,
    description: String
});

let Algorithm = mongoose.model("Algorithm", algorithmSchema);

module.exports = Algorithm;