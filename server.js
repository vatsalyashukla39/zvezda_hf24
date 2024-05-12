const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const path = require('path');

const app = express();
app.use(bodyParser.json()); // for parsing application/json
app.use(express.static(path.join(__dirname, 'public'))); // serve static files from 'public' directory

// Connect to MongoDB
mongoose.connect('mongodb+srv://notaaash:january@koach.vryq9e9.mongodb.net/', { useUnifiedTopology: true });

// Define a Mongoose schema for the coordinates
const coordinateSchema = new mongoose.Schema({}, { strict: false });

// Create a Mongoose model from the schema
const Coordinate = mongoose.model('Coordinate', coordinateSchema);

app.post('/json-data', (req, res) => {
  const coordinates = new Coordinate(req.body);

  coordinates.save((err, result) => {
    if (err) {
      console.error('Error inserting data into MongoDB', err);
      res.status(500).send('Error inserting data into MongoDB');
      return;
    }

    console.log("Coordinates inserted");
    res.send('Coordinates received and inserted');
  });
});

app.listen(3000, () => {
  console.log('Server is running on port 3000');
});