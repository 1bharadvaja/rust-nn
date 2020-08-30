#[macro_use]

extern crate ndarray;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::*;
use ndarray_rand::rand_distr::*;
use std::num;
use std::f64;
use std::convert::From;

// eventually making this more object oriented, so that you dont have to repeat the same lines over and over again
// will make Layer as a class, and everything underneath it

struct Layer_Dense {
	weights: Array2<f64>,
	biases: Array2<f64>,
	output: Option<Array2<f64>>,
}
impl Layer_Dense {
 	fn init(n_inputs: usize, n_neurons: usize) -> Self {
 		let weights =  Array2::random((n_inputs, n_neurons ), Uniform::new(-1.0, 1.0));
 		let biases = Array2::zeros((1, n_neurons));
		Layer_Dense{
			weights,
			biases,
			output: None,
		}
	}
	fn forward(&mut self, inputs: Array2<f64>){
		self.output = Some(inputs.dot(&self.weights) + &self.biases);
	}
}

#[derive(Debug)]
struct Activation_ReLU {
	output: Vec<f64>
}

impl Activation_ReLU {
	fn init() -> Self {
		let output = vec![0., 0., 0., 0., 0.];
		Activation_ReLU{
			output
		}
	}


	fn ActivationReLU(&mut self, inputs: Array2<f64>){
		self.output = inputs.iter().map(|x| x.max(0.0)).collect::<Vec<f64>>();
	}	
}

struct Sigmoid {
	output: Option<Vec<f64>>
}

impl Sigmoid {
	fn init() -> Self {
		let output: Vec<f64>;
		Sigmoid {
			output: None
		}
	}

	fn Sigmoid(&mut self, inputs: Array2<f64>) {
		self.output = Some(inputs.iter().map(|x| (1.0)/(1.0 + (-x).exp())).collect::<Vec<f64>>());
	}
}


struct SigmoidPrime{
	output: Option<f64>
}

impl SigmoidPrime {
	fn init(a: i32) -> Self {
		let output: Vec<f64>;
		Sigmoid {
			output: None
		}
	}

	fn SigmoidPrime(&mut self, inputs: Array2<f64>) {
		self.output = Some(inputs.iter().map(|x| ((-x).exp())/((1.0 + (-x).exp()).pow(2))).collect::<Vec<f64>>());
	}
}

struct cost{
	yhat: Array2<f64>,
	a: Option<Vec<f64>>,
	J: f32
}

impl cost {
	fn init() -> Self {
		let yhat: Vec<f64>;
		let J = 0.;
		cost {
			yhat: None,
			a: None,
			J: f32
		}
	}

	fn costFunction(&mut self, X: Array2<f64>, y: Array2<f64>) {
		self.yhat = self.forward(X);
		let a = &y - &self.yhat;
		let J = (a.iter().map(|x| x.pow(2)).sum())*0.5;

	}
}

fn main() {
	let x = arr2(&[[3., 5.],
			 		[5., 1.],
			 		[10., 2.]]);
	let y = arr2(&[[75.], 
			 		[82.], 
			 		[93.]]);
	let mut l1 = Layer_Dense::init(2, 3);
	l1.forward(x);
	let mut sig = Sigmoid::init(1);
	sig.Sigmoid(l1.output.unwrap());
	let l2in = Array::from_shape_vec((3, 3), sig.output.unwrap()); // L1 final output
//layer 2 begin
	let mut sig2 = Sigmoid::init(2);
	let mut l2 = Layer_Dense::init(3, 1);
	l2.forward(l2in.unwrap());
	sig2.Sigmoid(l2.output.unwrap());
	let finaloutput = Array::from_shape_vec((3, 1), sig2.output.unwrap());
//layer 2 end
	
//cost function
		




}
