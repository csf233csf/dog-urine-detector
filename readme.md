# Dog Urinate Detector

This is a project combining Arduino / Python Flask API / Machine Learning Model. 

Aim is to teach a dog the place/pad where it needs to urinate.

## Pipeline:

1. Movement sensors and water-level sensor data collect.

2. Server gets the real-time data. Model predicts the data to see if the dog needs to urinate.

3. If model predicts 1, the water-pump module "GET" request from server and pump the liquid.

4. Humidity and UV sensors detect dog urinating: triggers box to give treats and play owner's voice for positive reinforcement.

