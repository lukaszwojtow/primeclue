use primeclue::data::data_set::DataSet;
use primeclue::error::PrimeclueErr;
use primeclue::exec::score::Objective;
use primeclue::exec::training_group::TrainingGroup;
use primeclue::math::median;
use primeclue::user::Settings;
use std::ops::Add;
use std::time::Instant;

// Usage:
// cargo run --release --example mnist_fashion_median_check
fn main() -> Result<(), PrimeclueErr> {
    let count = 5;
    let mut results = Vec::with_capacity(count);
    for run in 0..count {
        if let Some(result) = check_once() {
            results.push(result);
            println!("Loop #{} of {} result: {} ", run + 1, count, result);
        } else {
            println!("No result for loop #{}", run + 1);
        }
    }
    if results.is_empty() {
        return Ok(());
    }
    let median = median(&mut results);
    println!("Median result: {}", median);
    Ok(())
}

fn check_once() -> Option<f32> {
    // Get default data path
    let data_path = Settings::new().unwrap().data_dir();

    // Read data from disk. Data must be in Primeclue's format, i.e. imported to `data.ssd` file.
    // Get training data
    let (training_data, verification_data) =
        DataSet::read_from_disk(&data_path.join("mnist_fashion_training"))
            .unwrap()
            .shuffle()
            .into_2_views_split();

    // Get training object that later will be used to get classifier. Third argument is objective that
    // we want to maximize for. Other types are accuracy (percentage) or cost.
    let mut training =
        TrainingGroup::new(training_data, verification_data, Objective::Accuracy, 10, &[])
            .ok()?;
    // Get some loop break condition. Here it's time limit, regardless of result or generation count
    let end_time = Instant::now().add(std::time::Duration::from_secs(60));

    // Actual training happens here
    while Instant::now().lt(&end_time) {
        training.next_generation();
    }
    // Get classifier after training has finished. It will fail if there is no classifier for any of the classes
    let classifier = training.classifier().ok()?;

    // Get testing data
    let testing_data =
        DataSet::read_from_disk(&data_path.join("mnist_fashion_testing")).unwrap().into_view();

    // Get classifier's score on unseen data
    Some(classifier.score(&testing_data)?.accuracy)
}
