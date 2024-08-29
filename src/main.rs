use lib::{activations::SIGMOID, mnist::MnistDataset, network::Network};

pub mod lib;

fn ocr_test() {
    let (train_images, train_labels, test_images, test_labels) = MnistDataset::preprocess_mnist(500, 100, 100);

    let mut network = Network::new(vec![784, 64, 10], 0.3, SIGMOID);

    // Train the network
    network.train(train_images, train_labels, 10);

    // Evaluate the network
    let mut correct = 0;
    for (image, label) in test_images.iter().zip(test_labels.iter()) {
        let output = network.feed_forward(image.clone());
        let predicted = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let actual = label
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        println!("predicted: {}, actual: {}", predicted, actual);
        if predicted == actual {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / test_images.len() as f64;
    println!("Accuracy: {:.2}%", accuracy * 100.0);
}

fn xor_test() {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let mut network = Network::new(vec![2, 5, 1], 0.5, SIGMOID);

    network.train(inputs.clone(), targets, 1000);

    for i in inputs {
        println!(
            "{} and {}: {:?}",
            i[0].clone(),
            i[1].clone(),
            network.feed_forward(i)
        );
    }
}

fn main() {
    ocr_test();
    xor_test();
}
