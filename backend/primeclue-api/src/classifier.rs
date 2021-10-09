// SPDX-License-Identifier: AGPL-3.0
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

use crate::executor::{Status, StatusCallback, Termination};
use primeclue::data::data_set::{DataSet, DataView, Rewards};
use primeclue::data::importer::{build_numbers_row, get_header_row, split_to_vec};
use primeclue::data::{Input, InputShape, Outcome, Point};
use primeclue::error::PrimeclueErr;
use primeclue::exec::classifier::{Classifier, ClassifierScore};
use primeclue::exec::score::Objective;
use primeclue::exec::training_group::{Stats, TrainingGroup};
use primeclue::serialization::serializator::SERIALIZED_FILE_EXT;
use primeclue::serialization::{Deserializable, Serializable, Serializator};
use primeclue::user::{read_files, Settings, CLASSIFIERS_DIR};
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::read_dir;
use std::ops::Add;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::time::Duration;

const CLASSIFIER_FILE_NAME: &str = "classifier.ssd";

#[derive(Deserialize, Debug)]
pub(crate) struct CreateRequest {
    data_name: String,
    classifier_name: String,
    training_objective: Objective,
    override_rewards: bool,
    rewards: Rewards,
    timeout: u64,
    size: usize,
    forbidden_columns: String,
    shuffle_data: bool,
    keep_unseen_data: bool,
}

pub(crate) fn create(
    request: &CreateRequest,
    status_callback: &StatusCallback,
    terminator: &Receiver<Termination>,
) -> Result<String, PrimeclueErr> {
    let data_set = read_data(request)?;
    let result = start_training(request, data_set, status_callback, terminator)?;
    Ok(result)
}

fn parse_forbidden_columns(text: &str) -> Result<Vec<usize>, PrimeclueErr> {
    let chunks = text.split(' ').filter(|&s| !s.is_empty()).collect::<Vec<_>>();
    let mut columns = Vec::new();
    for chunk in chunks {
        let column = chunk.trim().parse::<usize>().map_err(|e| {
            PrimeclueErr::from(format!("Unable to parse '{}' to column index: {}", chunk, e))
        })?;
        if column < 1 {
            return PrimeclueErr::result("Column indexing starts from 1".to_owned());
        }
        columns.push(column - 1);
    }
    Ok(columns)
}

fn print_cost_range(data1: &DataView, data2: &DataView) {
    let (max, min) = data1.cost_range();
    println!("Cost range for training data {} {}", min, max);
    let (max, min) = data2.cost_range();
    let random_guess = data2.random_guess_cost();
    println!("Cost range for test data {} {}, guessing randomly: {}", min, max, random_guess);
    let sum =
        data2.outcomes().iter().map(|outcome| outcome.reward() + outcome.penalty()).sum::<f32>();
    println!("Test data average cost / point: {}", sum / data2.outcomes().len() as f32);
}

fn start_training(
    request: &CreateRequest,
    mut data_set: DataSet,
    status_callback: &StatusCallback,
    terminator: &Receiver<Termination>,
) -> Result<String, PrimeclueErr> {
    if request.shuffle_data {
        data_set = data_set.shuffle();
    }
    let (training_data, verification_data, test_data) =
        split_into_sets(data_set, request.keep_unseen_data);
    print_cost_range(&training_data, &test_data);
    print_test_class_item_count(&test_data);
    let forbidden_cols = parse_forbidden_columns(&request.forbidden_columns)?;
    let dst_dir = create_classifier_dir(request)?;
    let mut training = TrainingGroup::new(
        training_data,
        verification_data,
        request.training_objective,
        request.size,
        &forbidden_cols,
    )?;
    let start_time = std::time::Instant::now();
    let end_time = start_time.add(Duration::from_secs(60 * request.timeout));
    while std::time::Instant::now().lt(&end_time) {
        if terminator.try_recv().is_ok() {
            if let Err(e) = fs::remove_dir_all(&dst_dir) {
                println!("Unable to remove classifier directory {:?}, error: {:?}", dst_dir, e);
            }
            return Ok("Terminating training for user request".to_string());
        }
        training.next_generation();
        if let Some(stats) = training.stats() {
            if let Ok(classifier) = training.classifier() {
                if let Some(classifier_score) = classifier.score(&test_data) {
                    let status = TrainingStatus { stats, classifier_score };
                    status_callback(Status::Progress(
                        0.0,
                        serde_json::to_string(&status).unwrap(),
                    ));
                }
            }
        }
    }
    save(&dst_dir, &training)?;
    Ok(format!(
        "Training finished with average score: {:?}",
        training.classifier()?.average_score()
    ))
}

fn print_test_class_item_count(test_data: &DataView) {
    println!("Test data label count:");
    for class in test_data.class_map().keys() {
        println!(
            "\t{}: {} items",
            test_data.class_map().get(class).unwrap(),
            test_data.class_items_count(*class).unwrap()
        );
    }
}

#[derive(Serialize)]
struct TrainingStatus {
    stats: Stats,
    classifier_score: ClassifierScore,
}

fn save(dst_dir: &Path, training: &TrainingGroup) -> Result<usize, PrimeclueErr> {
    let classifier = training.classifier()?;
    let mut s = Serializator::new();
    classifier.serialize(&mut s);
    s.save(dst_dir, CLASSIFIER_FILE_NAME).map_err(PrimeclueErr::from)
}

fn read_data(request: &CreateRequest) -> Result<DataSet, PrimeclueErr> {
    let settings = Settings::new()?;
    let src_data_dir = settings.data_dir().join(&request.data_name);
    let mut dsr = DataSet::read_from_disk(&src_data_dir)?;
    if request.override_rewards {
        dsr.apply_rewards(&request.rewards);
    }
    Ok(dsr)
}

pub(crate) fn list() -> Result<Vec<String>, PrimeclueErr> {
    let settings = Settings::new()?;
    let list = read_dir(&settings.classifier_dir())?;
    read_files(list)
}

pub(crate) fn remove(name: &str) -> Result<(), PrimeclueErr> {
    let settings = Settings::new()?;
    let path = Path::new(settings.base_dir()).join(CLASSIFIERS_DIR).join(name);
    fs::remove_dir_all(&path)?;
    Ok(())
}

fn create_classifier_dir(request: &CreateRequest) -> Result<PathBuf, PrimeclueErr> {
    let settings = Settings::new()?;
    let path =
        Path::new(settings.base_dir()).join(CLASSIFIERS_DIR).join(&request.classifier_name);
    if path.exists() {
        PrimeclueErr::result(format!("Path {:?} already exists", path))
    } else {
        fs::create_dir_all(&path)
            .map_err(|e| format!("Unable to create classifier dir: {:?}", e))?;
        Ok(path)
    }
}

#[derive(Deserialize, Debug)]
pub(crate) struct ClassifyRequest {
    classifier_name: String,
    content: String,
    separator: String,
    ignore_first_row: bool,
    data_columns: Vec<bool>,
}

impl ClassifyRequest {
    pub(crate) fn classify(&self) -> Result<String, PrimeclueErr> {
        let (classifiers, names) = self.read_classifiers()?;
        let header_row =
            get_header_row(&self.content, &self.separator, self.ignore_first_row, names);
        let mut data_raw = split_to_vec(&self.content, &self.separator, self.ignore_first_row);
        let numbers = parse_data(&data_raw, &self.data_columns)?;
        ClassifyRequest::validate_input_shape(&classifiers, &numbers)?;
        let responses_list = build_responses_list(&classifiers, &numbers);
        let mut classification = Vec::with_capacity(data_raw.len());
        classification.push(header_row.join(&self.separator));
        for r in 0..data_raw.len() {
            let row = &mut data_raw[r];
            for responses in &responses_list {
                let response = responses[r];
                row.push(response);
            }
            let line = row.join(self.separator.as_str());
            classification.push(line);
        }
        Ok(classification.join("\r\n"))
    }

    fn validate_input_shape(
        classifiers: &[Classifier],
        numbers: &[Vec<f32>],
    ) -> Result<(), PrimeclueErr> {
        for classifier in classifiers {
            check_size(numbers, classifier.input_shape())?;
        }
        Ok(())
    }

    fn read_classifiers(&self) -> Result<(Vec<Classifier>, Vec<String>), PrimeclueErr> {
        let settings = Settings::new()?;
        let mut classifiers = vec![];
        let path = settings.base_dir().join(CLASSIFIERS_DIR).join(&self.classifier_name);
        let mut names = vec![];
        for entry in read_dir(&path)? {
            let entry = entry?;
            let file_name = entry.file_name().into_string().unwrap();
            if file_name.ends_with(SERIALIZED_FILE_EXT) {
                names.push(file_name);
            }
        }
        names.sort();
        for name in &names {
            let path = path.join(&name);
            let classifier = Classifier::deserialize(&mut Serializator::load(&path)?)?;
            classifiers.push(classifier);
        }
        if classifiers.is_empty() {
            PrimeclueErr::result(format!("Unable to find serialized object in {:?}", path))
        } else {
            Ok((classifiers, names))
        }
    }
}

fn build_responses_list<'a>(
    classifiers: &'a [Classifier],
    numbers: &[Vec<f32>],
) -> Vec<Vec<&'a str>> {
    let mut responses_list = vec![];
    for classifier in classifiers {
        let responses = classify_all(numbers, classifier);
        responses_list.push(responses);
    }
    responses_list
}

fn split_into_sets(data: DataSet, keep_unseen: bool) -> (DataView, DataView, DataView) {
    if keep_unseen {
        data.into_3_views_split()
    } else {
        let test = data.clone().into_view();
        let (training, verification) = data.into_2_views_split();
        (training, verification, test)
    }
}

fn parse_data(raw: &[Vec<&str>], use_columns: &[bool]) -> Result<Vec<Vec<f32>>, PrimeclueErr> {
    let mut values: Vec<Vec<f32>> = Vec::with_capacity(raw.len());
    for (row_num, row) in raw.iter().enumerate() {
        let values_row = build_numbers_row(use_columns, row_num, row)?;
        if !values.is_empty() && values[0].len() != values_row.len() {
            return PrimeclueErr::result(format!(
                "Invalid {}'nth row length: found {}, expected {}",
                row_num,
                values_row.len(),
                values[0].len()
            ));
        }
        values.push(values_row);
    }
    Ok(values)
}

fn check_size(data: &[Vec<f32>], input_shape: &InputShape) -> Result<(), PrimeclueErr> {
    if data.len() < input_shape.rows() {
        PrimeclueErr::result(format!(
            "Invalid data size: not enough rows: is: {}, must be at least: {}",
            data.len(),
            input_shape.rows()
        ))
    } else if data[0].len() == input_shape.columns() {
        Ok(())
    } else {
        PrimeclueErr::result(format!(
            "Invalid data size: wrong number of columns: is: {}, must be: {}",
            data[0].len(),
            input_shape.columns()
        ))
    }
}

fn classify_all<'a>(numbers: &[Vec<f32>], classifier: &'a Classifier) -> Vec<&'a str> {
    let mut data_set = DataSet::new(classifier.get_classes().clone());
    for row in 0..=(numbers.len() - classifier.input_shape().rows()) {
        let input_data = build_input_data(numbers, row, classifier.input_shape());
        data_set.add_data_point(Point::new(input_data, Outcome::default())).unwrap();
    }
    let view = data_set.into_view();
    classifier.classify(&view)
}

fn build_input_data(numbers: &[Vec<f32>], row: usize, input_shape: &InputShape) -> Input {
    let mut data = vec![];
    let start_column = numbers[0].len() - input_shape.columns();
    for r in 0..input_shape.rows() {
        let numbers_row = &numbers[row + r];
        let last_columns = numbers_row.iter().skip(start_column).copied().collect();
        data.push(last_columns);
    }
    Input::from_vector(data).unwrap()
}
