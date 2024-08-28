use lib::{
    activations::SIGMOID,
    network::{self, Network},
};

pub mod lib;

fn main() {
    let network = Network::new(vec![2, 3, 1], 0.5, SIGMOID);
}
