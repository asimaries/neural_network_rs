use mnist::{Mnist, MnistBuilder};

pub struct MnistDataset;
impl MnistDataset {
    pub fn preprocess_mnist(
        training_set_size: u32,
        validation_set_size: u32,
        test_set_size: u32,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(training_set_size)
            .validation_set_length(validation_set_size)
            .test_set_length(test_set_size)
            .finalize();

        let train_images = trn_img
            .chunks(28 * 28)
            .map(|chunk| chunk.iter().map(|&pixel| pixel as f64 / 255.0).collect())
            .collect::<Vec<Vec<f64>>>();

        let train_labels = trn_lbl
            .iter()
            .map(|&label| {
                let mut target = vec![0.0; 10];
                target[label as usize] = 1.0;
                target
            })
            .collect::<Vec<Vec<f64>>>();

        let test_images = tst_img
            .chunks(28 * 28)
            .map(|chunk| chunk.iter().map(|&pixel| pixel as f64 / 255.0).collect())
            .collect::<Vec<Vec<f64>>>();

        let test_labels = tst_lbl
            .iter()
            .map(|&label| {
                let mut target = vec![0.0; 10];
                target[label as usize] = 1.0;
                target
            })
            .collect::<Vec<Vec<f64>>>();

        (train_images, train_labels, test_images, test_labels)
    }
}
