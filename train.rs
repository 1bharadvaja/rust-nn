use ndarray::{Array2, Array1, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

#[derive(Debug)]
struct MLP {
    layers: Vec<Array2<f64>>, // Weights for each layer
    biases: Vec<Array1<f64>>, // Biases for each layer
}

impl MLP {
    fn new(layer_sizes: &[usize]) -> Self {
        let mut rng = rand::thread_rng();
        let mut layers = Vec::new();
        let mut biases = Vec::new();

        for pair in layer_sizes.windows(2) {
            // Initialize weights and biases
            layers.push(Array2::random((pair[1], pair[0]), Uniform::new(-0.5, 0.5)));
            biases.push(Array1::random(pair[1], Uniform::new(-0.5, 0.5)));
        }

        MLP { layers, biases }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_derivative(x: f64) -> f64 {
        x * (1.0 - x)
    }

    fn forward(&self, input: &Array1<f64>) -> Vec<Array1<f64>> {
        let mut activations = vec![input.clone()];
        let mut current_input = input.clone();

        for (weights, biases) in self.layers.iter().zip(&self.biases) {
            let z = weights.dot(&current_input) + biases;
            let a = z.mapv(Self::sigmoid);
            activations.push(a.clone());
            current_input = a;
        }

        activations
    }

    fn backpropagation(&self, activations: &[Array1<f64>], expected: &Array1<f64>) -> (Vec<Array2<f64>>, Vec<Array1<f64>>) {
        let mut layer_errors = Vec::new();
        let mut layer_deltas = Vec::new();

        // Compute output error
        let output_error = &activations.last().unwrap() - expected;
        layer_errors.push(output_error.clone());

        // Backpropagate
        for (i, weights) in self.layers.iter().enumerate().rev() {
            let error = if i == self.layers.len() - 1 {
                &output_error
            } else {
                weights.t().dot(&layer_errors.last().unwrap())
            };

            let delta = error * activations[i + 1].mapv(Self::sigmoid_derivative);
            layer_errors.push(error);
            layer_deltas.push(delta);
        }

        layer_deltas.reverse();
        let mut weight_gradients = Vec::new();
        let mut bias_gradients = Vec::new();

        for (i, delta) in layer_deltas.iter().enumerate() {
            weight_gradients.push(delta.insert_axis(Axis(1)).dot(&activations[i].insert_axis(Axis(0))));
            bias_gradients.push(delta.clone());
        }

        (weight_gradients, bias_gradients)
    }

    fn train(&mut self, inputs: Vec<Array1<f64>>, targets: Vec<Array1<f64>>, learning_rate: f64, epochs: usize) {
        for _ in 0..epochs {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let activations = self.forward(input);
                let (weight_gradients, bias_gradients) = self.backpropagation(&activations, target);

                // Update weights and biases
                for (i, weights) in self.layers.iter_mut().enumerate() {
                    *weights -= &(weight_gradients[i] * learning_rate);
                }

                for (i, biases) in self.biases.iter_mut().enumerate() {
                    *biases -= &(bias_gradients[i] * learning_rate);
                }
            }
        }
    }

    fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        self.forward(input).last().unwrap().clone()
    }
}

fn main() {
    // Define a simple MLP: 784 -> 128 -> 64 -> 10 (e.g., for MNIST classification)
    let mut mlp = MLP::new(&[784, 128, 64, 10]);

    // Simulated data (replace with real MNIST dataset loading)
    let input = Array1::random(784, Uniform::new(0.0, 1.0));
    let target = Array1::zeros(10);

    // Train the model (dummy example)
    mlp.train(vec![input.clone()], vec![target.clone()], 0.01, 10);

    // Predict
    let prediction = mlp.predict(&input);
    println!("Prediction: {:?}", prediction);
}
