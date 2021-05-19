// SPDX-License-Identifier: AGPL-3.0-or-later
/*
   Primeclue: Machine Learning and Data Mining
   Copyright (C) 2020 Łukasz Wojtów

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as
   published by the Free Software Foundation, either version 3 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

use primeclue::data::data_set::DataSet;
use primeclue::data::outcome::Class;
use primeclue::data::{Input, Outcome, Point};
use primeclue::exec::score::Objective;
use primeclue::exec::training_group::TrainingGroup;
use primeclue::rand::GET_RNG;
use rand::Rng;
use std::collections::HashMap;

// A difficult-ish problem for ML. Not only test data is never seen during training, but it also
// have values in range > 200 && <= 300. Values in this range were not observed during training.

// Run with : cargo run --release --example test_training
// Average score on unseen data: 0.66
fn main() {
    let mut sum = 0.0;
    let count = 100;
    for attempt in 1..count + 1 {
        sum += training_success(attempt);
        println!(
            "Average score on unseen data after {} attempts: {}",
            attempt,
            sum / attempt as f32
        );
    }
}

fn training_success(attempt: usize) -> f32 {
    let mut classes = HashMap::new();
    classes.insert(Class::new(0), "A".to_owned());
    classes.insert(Class::new(1), "B".to_owned());
    classes.insert(Class::new(2), "C".to_owned());
    classes.insert(Class::new(3), "D".to_owned());
    let string_classes = classes.iter().map(|(c, s)| (s.clone(), *c)).collect::<HashMap<_, _>>();
    let mut data_set = DataSet::new(classes);
    let mut rng = GET_RNG();
    let max = 100;
    for i in 0..3 {
        for _ in 0..1_000 {
            let a = i * max + rng.gen_range(0, max);
            let b = i * max + rng.gen_range(0, max);
            let c = i * max + rng.gen_range(0, max);
            let output = if a % 15 == 0 {
                "A"
            } else if (b + 2) % 5 == 0 {
                "B"
            } else if (c + 5) % 3 == 0 {
                "C"
            } else {
                "D"
            };
            let point = Point::new(
                Input::from_vector(vec![vec![a as f32, b as f32, c as f32]]).unwrap(),
                Outcome::new(*string_classes.get(output).unwrap(), 1.0, -1.0),
            );
            data_set.add_data_point(point).unwrap();
        }
    }

    let (training_data, verification_data, test_data) = data_set.into_3_views_split();

    let mut training = TrainingGroup::new(
        training_data,
        verification_data,
        &Objective::Accuracy,
        100,
        &Vec::new(),
    )
    .unwrap();
    loop {
        training.next_generation();
        if let Some(stats) = training.stats() {
            if stats.generation > 10_000 {
                println!("Training failed! Unable to learn after 10k generations");
                std::process::exit(1);
            }
            if let Ok(classifier) = training.classifier() {
                if let Some(score) = classifier.score(&test_data) {
                    if let Some(stats) = training.stats() {
                        println!(
                            "Testing training #{}, epoch: {}, training: {:4.2}, unseen: {:4.2}",
                            attempt, stats.generation, stats.training_score, score.accuracy
                        );
                        if stats.training_score >= 0.9 {
                            println!(
                                "Testing training #{}, epoch: {}, training: {:4.2}, unseen: {:4.2}",
                                attempt, stats.generation, stats.training_score, score.accuracy
                            );
                            return score.accuracy;
                        }
                    }
                }
            }
        }
    }
}
